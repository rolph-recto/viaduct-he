use super::HEProgram;

pub mod pyseal;

/// backend that generates code from an HEProgram.
pub trait HEBackend<'a> {
    fn name(&self) -> &str;

    fn compile(
        &mut self,
        program: HEProgram,
        writer: Box<dyn std::io::Write + 'a>,
    ) -> std::io::Result<()>;
}

// "dummy" backend that just writes the HEProgram
pub struct DummyBackend;

impl<'a> HEBackend<'a> for DummyBackend {
    fn name(&self) -> &str { "dummy" }

    fn compile(
        &mut self,
        program: HEProgram,
        mut writer: Box<dyn std::io::Write + 'a>,
    ) -> std::io::Result<()> {
        program.to_doc().render(80, &mut (*writer))
    }
}
