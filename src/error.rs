use rulinalg;

#[derive(Debug)]
pub enum Error {
    RulinalgError(rulinalg::error::Error),
    UnknownError,
}

impl From<rulinalg::error::Error> for Error {
    fn from(e: rulinalg::error::Error) -> Error {
        Error::RulinalgError(e)
    }
}
