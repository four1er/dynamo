// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use tonic_health_v14 as tonic_health;
use tonic_v14 as tonic;

use std::future::Future;
use std::time::Duration;

use dynamo_backend_common::{BackendError, DynamoError, ErrorType};
use dynamo_sidecar_common::v14::GrpcChannelPool;
use dynamo_sidecar_common::{DEFAULT_MAX_GRPC_MESSAGE_SIZE, GrpcEndpoint, GrpcTransportConfig};
use tokio::time::{Instant, sleep_until, timeout_at};
use tonic::metadata::MetadataValue;
use tonic::transport::Channel;
use tonic_health::pb::health_check_response::ServingStatus;
use tonic_health::pb::{HealthCheckRequest, health_client::HealthClient};

pub(crate) use dynamo_sidecar_common::v14::{status_to_dynamo, stream_status_to_dynamo};
pub(crate) use dynamo_sidecar_common::{
    engine_shutdown, invalid_argument, startup_deadline, stream_incomplete,
};

use crate::proto as pb;

pub(crate) const CONTROL_SERVICE: &str = "vllm.Control";
pub(crate) const INFERENCE_SERVICE: &str = "vllm.Inference";
const DATA_PARALLEL_RANK_METADATA_KEY: &str = "x-data-parallel-rank";

pub(crate) struct VllmClient {
    pool: GrpcChannelPool,
}

impl VllmClient {
    /// `bootstrap`: see `GrpcChannelPool::connect`. True for pre-logging
    /// synchronous constructors; false for deferred discovery and engine start.
    pub(crate) async fn connect(
        endpoint: &GrpcEndpoint,
        transport: GrpcTransportConfig,
        startup_deadline: Instant,
        bootstrap: bool,
    ) -> Result<Self, DynamoError> {
        let pool = timeout_at(
            startup_deadline,
            GrpcChannelPool::connect("vLLM", endpoint, transport, bootstrap),
        )
        .await
        .map_err(|_| {
            dynamo_sidecar_common::connection_timeout(format!(
                "vLLM gRPC connection pool to {endpoint} exceeded the total startup deadline"
            ))
        })??;
        Ok(Self { pool })
    }

    pub(crate) fn connection_count(&self) -> usize {
        self.pool.len()
    }

    pub(crate) fn health_channel(&self) -> Channel {
        self.pool.next_channel()
    }

    pub(crate) fn control_client(&self) -> pb::control_client::ControlClient<Channel> {
        pb::control_client::ControlClient::new(self.pool.next_channel())
            .max_encoding_message_size(DEFAULT_MAX_GRPC_MESSAGE_SIZE)
            .max_decoding_message_size(DEFAULT_MAX_GRPC_MESSAGE_SIZE)
    }

    pub(crate) async fn wait_for_services(
        &self,
        services: &[&str],
        startup_deadline: Instant,
        retry_interval: Duration,
    ) -> Result<(), DynamoError> {
        for service in services {
            self.wait_for_service(service, startup_deadline, retry_interval)
                .await?;
        }
        Ok(())
    }

    async fn wait_for_service(
        &self,
        service: &str,
        deadline: Instant,
        retry_interval: Duration,
    ) -> Result<(), DynamoError> {
        let started = Instant::now();
        loop {
            let mut client = HealthClient::new(self.pool.next_channel());
            let last_status = match timeout_at(
                deadline,
                client.check(HealthCheckRequest {
                    service: service.to_string(),
                }),
            )
            .await
            {
                Ok(Ok(response)) => {
                    let status = ServingStatus::try_from(response.into_inner().status)
                        .unwrap_or(ServingStatus::Unknown);
                    if status == ServingStatus::Serving {
                        return Ok(());
                    }
                    format!("reported {}", status.as_str_name())
                }
                Ok(Err(status))
                    if matches!(
                        status.code(),
                        tonic::Code::NotFound | tonic::Code::Unimplemented
                    ) =>
                {
                    return Err(protocol_error(format!(
                        "{service} is unavailable through the standard gRPC health API: {status}"
                    )));
                }
                Ok(Err(status)) => format!("health check failed: {status}"),
                Err(_) => "health check exceeded the startup deadline".to_string(),
            };

            let now = Instant::now();
            if now >= deadline {
                return Err(dynamo_sidecar_common::cannot_connect(format!(
                    "{service} did not become SERVING before the total vLLM startup deadline after {:?}: {last_status}",
                    started.elapsed()
                )));
            }
            let retry_at = now.checked_add(retry_interval).unwrap_or(deadline);
            sleep_until(retry_at.min(deadline)).await;
        }
    }

    pub(crate) async fn discover(
        &self,
        startup_deadline: Instant,
    ) -> Result<(pb::ModelInfo, pb::ServerInfo), DynamoError> {
        let mut client = self.control_client();
        let model = timeout_at(
            startup_deadline,
            client.get_model_info(pb::GetModelInfoRequest {}),
        )
        .await
        .map_err(|_| {
            dynamo_sidecar_common::connection_timeout(
                "GetModelInfo exceeded the total vLLM startup deadline",
            )
        })?
        .map(tonic::Response::into_inner)
        .map_err(|status| status_to_dynamo("GetModelInfo", status))?;
        let server = timeout_at(
            startup_deadline,
            client.get_server_info(pb::GetServerInfoRequest {}),
        )
        .await
        .map_err(|_| {
            dynamo_sidecar_common::connection_timeout(
                "GetServerInfo exceeded the total vLLM startup deadline",
            )
        })?
        .map(tonic::Response::into_inner)
        .map_err(|status| status_to_dynamo("GetServerInfo", status))?;
        Ok((model, server))
    }

    pub(crate) async fn generate_stream(
        &self,
        request: pb::GenerateRequest,
        data_parallel_rank: Option<u32>,
    ) -> Result<tonic::Streaming<pb::GenerateResponse>, DynamoError> {
        let mut client = pb::inference_client::InferenceClient::new(self.pool.next_channel())
            .max_encoding_message_size(DEFAULT_MAX_GRPC_MESSAGE_SIZE)
            .max_decoding_message_size(DEFAULT_MAX_GRPC_MESSAGE_SIZE);
        let mut request = tonic::Request::new(request);
        if let Some(rank) = data_parallel_rank {
            request
                .metadata_mut()
                .insert(DATA_PARALLEL_RANK_METADATA_KEY, MetadataValue::from(rank));
        }
        client
            .generate_stream(request)
            .await
            .map(tonic::Response::into_inner)
            .map_err(|status| generation_status_to_dynamo("GenerateStream", status))
    }

    pub(crate) async fn kv_event_sources(&self) -> Result<Vec<pb::KvEventSource>, DynamoError> {
        let mut client = pb::control_client::ControlClient::new(self.pool.next_channel())
            .max_encoding_message_size(DEFAULT_MAX_GRPC_MESSAGE_SIZE)
            .max_decoding_message_size(DEFAULT_MAX_GRPC_MESSAGE_SIZE);
        client
            .get_kv_event_sources(pb::GetKvEventSourcesRequest {})
            .await
            .map(tonic::Response::into_inner)
            .map(|response| response.sources)
            .map_err(|status| status_to_dynamo("GetKvEventSources", status))
    }

    /// Abort by caller request ID. vLLM's GenerateStream producer does not
    /// watch for a closed client stream, so dropping the stream alone leaves a
    /// queued request in the engine until a slot frees up.
    pub(crate) async fn abort(&self, request_id: String) -> Result<(), DynamoError> {
        let mut client = self.control_client();
        tokio::time::timeout(
            ABORT_RPC_DEADLINE,
            client.abort(pb::AbortRequest {
                request_ids: vec![request_id],
            }),
        )
        .await
        .map_err(|_| {
            dynamo_sidecar_common::connection_timeout(format!(
                "Abort did not respond within {ABORT_RPC_DEADLINE:?}"
            ))
        })?
        .map(|_| ())
        .map_err(|status| status_to_dynamo("Abort", status))
    }

    pub(crate) async fn load_lora(
        &self,
        lora_name: String,
        source_path: String,
    ) -> Result<pb::LoadLoraResponse, LoraRpcError> {
        let mut client = self.control_client();
        lora_rpc(
            "LoadLora",
            client.load_lora(pb::LoadLoraRequest {
                lora_name,
                source_path,
            }),
        )
        .await
    }

    pub(crate) async fn unload_lora(
        &self,
        lora_name: String,
    ) -> Result<pb::UnloadLoraResponse, LoraRpcError> {
        let mut client = self.control_client();
        lora_rpc(
            "UnloadLora",
            client.unload_lora(pb::UnloadLoraRequest { lora_name }),
        )
        .await
    }

    pub(crate) async fn list_loras(&self) -> Result<Vec<pb::LoraAdapter>, LoraRpcError> {
        let mut client = self.control_client();
        lora_rpc("ListLoras", client.list_loras(pb::ListLorasRequest {}))
            .await
            .map(|response| response.adapters)
    }
}

pub(crate) fn protocol_error(message: impl Into<String>) -> DynamoError {
    dynamo_sidecar_common::protocol_error("vLLM", message)
}

/// Follows vLLM's overall gRPC health (`""`) until vLLM stops serving, and
/// returns why. vLLM changes it only on engine failure or shutdown, never while
/// paused or asleep.
pub(crate) async fn watch_health(channel: Channel, open_timeout: Duration) -> String {
    let request = HealthCheckRequest {
        service: String::new(),
    };
    let mut stream =
        match tokio::time::timeout(open_timeout, HealthClient::new(channel).watch(request)).await {
            Ok(Ok(response)) => response.into_inner(),
            Ok(Err(status)) => return format!("health watch failed: {status}"),
            Err(_) => return format!("health watch did not open in {open_timeout:?}"),
        };
    loop {
        match stream.message().await {
            Ok(Some(response)) => {
                let status =
                    ServingStatus::try_from(response.status).unwrap_or(ServingStatus::Unknown);
                if status != ServingStatus::Serving {
                    return format!("health reported {}", status.as_str_name());
                }
            }
            Ok(None) => return "health stream ended".to_string(),
            Err(status) => return format!("health stream failed: {status}"),
        }
    }
}

// Bound how long a stalled RPC can hold the lifecycle lock.
pub(crate) const LORA_RPC_DEADLINE: Duration = Duration::from_secs(60);

const ABORT_RPC_DEADLINE: Duration = Duration::from_secs(30);

#[derive(Debug, Clone)]
pub(crate) struct LoraRpcError {
    pub(crate) rpc: &'static str,
    pub(crate) code: tonic::Code,
    pub(crate) message: String,
}

impl LoraRpcError {
    // Other errors may follow a committed mutation; reconcile them with ListLoras.
    pub(crate) fn is_definitive(&self) -> bool {
        matches!(
            self.code,
            tonic::Code::InvalidArgument
                | tonic::Code::AlreadyExists
                | tonic::Code::NotFound
                | tonic::Code::FailedPrecondition
        )
    }

    pub(crate) fn into_dynamo(self) -> DynamoError {
        status_to_dynamo(self.rpc, tonic::Status::new(self.code, self.message))
    }
}

impl std::fmt::Display for LoraRpcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {} ({:?})", self.rpc, self.message, self.code)
    }
}

async fn lora_rpc<T>(
    rpc: &'static str,
    call: impl Future<Output = Result<tonic::Response<T>, tonic::Status>>,
) -> Result<T, LoraRpcError> {
    match tokio::time::timeout(LORA_RPC_DEADLINE, call).await {
        Ok(Ok(response)) => Ok(response.into_inner()),
        Ok(Err(status)) => Err(LoraRpcError {
            rpc,
            code: status.code(),
            message: status.message().to_string(),
        }),
        Err(_) => Err(LoraRpcError {
            rpc,
            code: tonic::Code::DeadlineExceeded,
            message: format!("{rpc} exceeded the {LORA_RPC_DEADLINE:?} lifecycle deadline"),
        }),
    }
}

const EMPTY_GRAMMAR: &str = "Invalid grammar: grammar must not be empty";
const EMPTY_SCHEMA: &str = "Invalid JSON schema: schema must not be empty";
const EMPTY_ENUM: &str = "Invalid JSON schema: enum array must not be empty";
const UNSATISFIABLE_SCHEMA: &str = "Invalid JSON schema: a required property is unsatisfiable";
const INVALID_SCHEMA: &str = "Invalid JSON schema";
const INVALID_CONSTRAINT: &str = "Invalid structured-output constraint";

/// Map a generate-stream status. Recognized structured-output failures get an
/// authored public reason; the gRPC text stays diagnostic so compiler paths and
/// unknown statuses cannot reach the client.
pub(crate) fn generation_status_to_dynamo(rpc: &str, status: tonic::Status) -> DynamoError {
    let public = structured_output_public_reason(status.code(), status.message());
    let mut error = stream_status_to_dynamo(rpc, status);
    if let Some(message) = public
        && error.error_type() == ErrorType::Backend(BackendError::InvalidArgument)
    {
        error.public = Some(dynamo_runtime::error::PublicDetails::Message {
            message: message.to_owned(),
        });
    }
    error
}

fn structured_output_public_reason(code: tonic::Code, message: &str) -> Option<&'static str> {
    if code != tonic::Code::InvalidArgument {
        return None;
    }
    let (compiler_log, stripped) = without_compiler_logs(message);
    let text = stripped.trim();
    if text.contains("structured_outputs.grammar cannot be an empty string") {
        return Some(EMPTY_GRAMMAR);
    }
    if text.contains("structured_outputs.json cannot be an empty string") {
        return Some(EMPTY_SCHEMA);
    }
    if text == "enum array must not be empty" || text.ends_with(": enum array must not be empty") {
        return Some(EMPTY_ENUM);
    }
    if text.contains("Unsatisfiable schema: required property '")
        && text.ends_with("' is unsatisfiable")
    {
        return Some(UNSATISFIABLE_SCHEMA);
    }
    if text.contains("Invalid JSON grammar specification.") || text.contains("invalid json schema:")
    {
        return Some(INVALID_SCHEMA);
    }
    if compiler_log
        || text.contains("Failed to transform json schema into a grammar:")
        || text.contains("Grammar error:")
        || recognized_structured_output_sentence(text)
    {
        return Some(INVALID_CONSTRAINT);
    }
    None
}

fn recognized_structured_output_sentence(text: &str) -> bool {
    [
        "structured_outputs.choice cannot be empty",
        "must not contain a NUL character",
        "structural_tag cannot be an empty string",
        "Invalid structural tag specification.",
        "features not supported by xgrammar",
        "does not support json_object",
        "does not support grammar specifications",
        "does not support structural_tag",
        "only supports structural tags with `structures` and `triggers`",
        "nested too deeply",
        "json_object must be true if set",
        "multiple structured output constraints specified",
        "missing structured output constraint",
    ]
    .iter()
    .any(|sentence| text.contains(sentence))
}

/// Strip vLLM/xgrammar locations of the form `[HH:MM:SS] /path/file.cc:1: `.
fn without_compiler_logs(message: &str) -> (bool, &str) {
    let mut rest = message;
    let mut found = false;
    while let Some(stripped) = strip_compiler_log_prefix(rest) {
        found = true;
        rest = stripped;
    }
    (found, rest.trim())
}

fn strip_compiler_log_prefix(message: &str) -> Option<&str> {
    let rest = message.trim_start();
    let bytes = rest.as_bytes();
    if bytes.len() < 12
        || bytes[0] != b'['
        || bytes[3] != b':'
        || bytes[6] != b':'
        || bytes[9] != b']'
        || bytes[10] != b' '
    {
        return None;
    }
    if !bytes[1..3]
        .iter()
        .chain(&bytes[4..6])
        .chain(&bytes[7..9])
        .all(u8::is_ascii_digit)
    {
        return None;
    }
    let after = rest.get(11..)?;
    let (location, tail) = after.split_once(": ")?;
    let (path, line) = location.rsplit_once(':')?;
    if path.is_empty() || path.chars().any(char::is_whitespace) {
        return None;
    }
    if line.is_empty() || !line.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    Some(tail)
}

#[cfg(test)]
mod tests {
    use dynamo_backend_common::{BackendError, ErrorType};
    use dynamo_llm::generate_format::WorkerError;

    use super::{
        EMPTY_ENUM, EMPTY_GRAMMAR, EMPTY_SCHEMA, INVALID_CONSTRAINT, INVALID_SCHEMA,
        UNSATISFIABLE_SCHEMA, generation_status_to_dynamo,
    };

    fn invalid(message: &str) -> tonic_v14::Status {
        tonic_v14::Status::new(tonic_v14::Code::InvalidArgument, message)
    }

    fn reason(message: &str) -> Option<String> {
        generation_status_to_dynamo("GenerateStream", invalid(message))
            .public_message()
            .map(str::to_owned)
    }

    #[test]
    fn recognized_grammar_rejection_publishes_a_safe_reason() {
        let raw = "structured_outputs.grammar cannot be an empty string";
        let error = generation_status_to_dynamo("GenerateStream", invalid(raw));
        assert_eq!(
            error.error_type(),
            ErrorType::Backend(BackendError::InvalidArgument)
        );
        assert_eq!(error.public_message(), Some(EMPTY_GRAMMAR));
        assert!(error.to_string().contains(raw));
        assert!(error.to_string().contains("InvalidArgument"));

        let encoded = serde_json::to_vec(&error).unwrap();
        let decoded: dynamo_backend_common::DynamoError = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded.public_message(), Some(EMPTY_GRAMMAR));
        let surfaced = WorkerError::canonical_invalid_request_message(&decoded).unwrap();
        assert_eq!(surfaced, EMPTY_GRAMMAR);
        assert!(!surfaced.contains("structured_outputs"));
    }

    #[test]
    fn recognized_schema_sentences_use_authored_reasons() {
        assert_eq!(
            reason("structured_outputs.json cannot be an empty string").as_deref(),
            Some(EMPTY_SCHEMA)
        );
        assert_eq!(
            reason("validator: enum array must not be empty").as_deref(),
            Some(EMPTY_ENUM)
        );
        assert_eq!(
            reason(
                "Grammar error: Unsatisfiable schema: required property 'status' is unsatisfiable"
            )
            .as_deref(),
            Some(UNSATISFIABLE_SCHEMA)
        );
        let schema = reason("invalid json schema: expected value at line 1 column 1").unwrap();
        assert_eq!(schema, INVALID_SCHEMA);
        assert!(!schema.contains("line 1"));
        assert_eq!(
            reason("Invalid JSON grammar specification.").as_deref(),
            Some(INVALID_SCHEMA)
        );
        assert_eq!(
            reason("structured_outputs.choice cannot be empty").as_deref(),
            Some(INVALID_CONSTRAINT)
        );
    }

    #[test]
    fn compiler_locations_stay_private() {
        let raw = "[07:13:49] /tmp/private/compiler.cc:1: Failed to transform json schema into a grammar: boom";
        let error = generation_status_to_dynamo("GenerateStream", invalid(raw));
        let public = error.public_message().unwrap();
        assert_eq!(public, INVALID_CONSTRAINT);
        assert!(!public.contains("/tmp"));
        assert!(!public.contains("compiler.cc"));
        assert!(!public.contains("boom"));
        assert!(error.to_string().contains("/tmp/private/compiler.cc"));
        let encoded = serde_json::to_string(&error).unwrap();
        let decoded: dynamo_backend_common::DynamoError = serde_json::from_str(&encoded).unwrap();
        let surfaced = WorkerError::canonical_invalid_request_message(&decoded).unwrap();
        assert_eq!(surfaced, INVALID_CONSTRAINT);
        assert!(!surfaced.contains("compiler.cc"));
    }

    #[test]
    fn unknown_invalid_argument_and_other_statuses_stay_unchanged() {
        let unknown = generation_status_to_dynamo(
            "GenerateStream",
            invalid("token 7 missing from /tmp/secret/model.json"),
        );
        assert_eq!(unknown.public_message(), None);
        assert!(unknown.to_string().contains("/tmp/secret/model.json"));
        assert_eq!(
            WorkerError::canonical_invalid_request_message(&unknown).as_deref(),
            Some("Invalid request")
        );

        let cancelled = generation_status_to_dynamo(
            "GenerateStream",
            tonic_v14::Status::new(
                tonic_v14::Code::Cancelled,
                "structured_outputs.grammar cannot be an empty string",
            ),
        );
        assert_eq!(
            cancelled.error_type(),
            ErrorType::Backend(BackendError::Cancelled)
        );
        assert_eq!(cancelled.public_message(), None);

        let broken = tonic_v14::Status::from_error(Box::new(std::io::Error::new(
            std::io::ErrorKind::ConnectionReset,
            "structured_outputs.grammar cannot be an empty string",
        )));
        let disconnected = generation_status_to_dynamo("GenerateStream", broken);
        assert_eq!(
            disconnected.error_type(),
            ErrorType::Backend(BackendError::Disconnected)
        );
        assert_eq!(disconnected.public_message(), None);
    }
}
