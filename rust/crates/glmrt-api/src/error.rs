use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Serialize;

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: ErrorBody,
}

#[derive(Debug, Serialize)]
struct ErrorBody {
    message: String,
    #[serde(rename = "type")]
    error_type: &'static str,
    param: Option<String>,
    code: Option<String>,
}

#[derive(Debug)]
pub(crate) struct ApiError {
    pub(crate) status: StatusCode,
    pub(crate) message: String,
    pub(crate) param: Option<String>,
    pub(crate) code: Option<String>,
}

impl ApiError {
    pub(crate) fn into_response(self) -> Response {
        openai_error(self.status, self.message, self.param, self.code)
    }
}

pub(crate) fn invalid_request(
    message: impl Into<String>,
    param: Option<impl Into<String>>,
) -> ApiError {
    ApiError {
        status: StatusCode::BAD_REQUEST,
        message: message.into(),
        param: param.map(Into::into),
        code: Some("invalid_request".to_owned()),
    }
}

pub(crate) fn runtime_error(message: impl std::fmt::Display) -> ApiError {
    ApiError {
        status: StatusCode::BAD_GATEWAY,
        message: message.to_string(),
        param: None,
        code: Some("backend_error".to_owned()),
    }
}

pub(crate) fn openai_error(
    status: StatusCode,
    message: String,
    param: Option<String>,
    code: Option<String>,
) -> Response {
    (
        status,
        Json(ErrorResponse {
            error: ErrorBody {
                message,
                error_type: "invalid_request_error",
                param,
                code,
            },
        }),
    )
        .into_response()
}
