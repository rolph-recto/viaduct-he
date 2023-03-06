use super::HEProgram;

pub mod pyseal;

/// backend that generates code from an HEProgram.
pub trait HEBackend {
    fn compile(
        self,
        program: HEProgram,
        writer: &mut impl std::fmt::Write
    ) -> std::fmt::Result;
}
