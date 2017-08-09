use rulinalg;

#[derive(Debug, Clone)]
pub struct Error {}

impl From<rulinalg::error::Error> for Error {
    fn from(_: rulinalg::error::Error) -> Error {
        Error {}
    }
}