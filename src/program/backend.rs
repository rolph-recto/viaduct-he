pub mod pyseal;

/// backend that generates code from an HEProgram.
pub trait HEBackend {
    fn compile(&mut self, program: HEProgram, writer: std::fmt::Write);
}
