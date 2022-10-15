use std::fs::File;

use handlebars::{Handlebars, handlebars_helper, RenderError};

use super::lowered_program::{HELoweredInstr, HELoweredProgram};

handlebars_helper!(instr_is_binary: |instr: HELoweredInstr| match instr {
    HELoweredInstr::Add { id: _, op1: _, op2: _} => true,
    HELoweredInstr::AddInplace { op1: _, op2: _} => true,
    HELoweredInstr::AddPlain { id: _, op1: _, op2: _ } => true,
    HELoweredInstr::AddPlainInplace { op1: _, op2: _ } => true,
    HELoweredInstr::Sub { id: _, op1: _, op2: _} => true,
    HELoweredInstr::SubInplace { op1: _, op2: _} => true,
    HELoweredInstr::SubPlain { id: _, op1: _, op2: _ } => true,
    HELoweredInstr::SubPlainInplace { op1: _, op2: _ } => true,
    HELoweredInstr::Negate { id: _, op1: _ } => false,
    HELoweredInstr::NegateInplace { op1: _ } => false,
    HELoweredInstr::Mul { id: _, op1: _, op2: _ } => true,
    HELoweredInstr::MulInplace { op1: _, op2: _ } => true,
    HELoweredInstr::MulPlain { id: _, op1: _, op2: _ } => true,
    HELoweredInstr::MulPlainInplace { op1: _, op2: _ } => true,
    HELoweredInstr::Rot { id: _, op1: _, op2: _ } => true,
    HELoweredInstr::RotInplace { op1: _, op2: _ } => true,
    HELoweredInstr::RelinearizeInplace { op1: _ } => false
});

handlebars_helper!(instr_is_inplace: |instr: HELoweredInstr| match instr {
    HELoweredInstr::Add { id: _, op1: _, op2: _} => false,
    HELoweredInstr::AddInplace { op1: _, op2: _} => true,
    HELoweredInstr::AddPlain { id: _, op1: _, op2: _ } => false,
    HELoweredInstr::AddPlainInplace { op1: _, op2: _ } => true,
    HELoweredInstr::Sub { id: _, op1: _, op2: _} => false,
    HELoweredInstr::SubInplace { op1: _, op2: _} => true,
    HELoweredInstr::SubPlain { id: _, op1: _, op2: _ } => false,
    HELoweredInstr::SubPlainInplace { op1: _, op2: _ } => true,
    HELoweredInstr::Negate { id: _, op1: _ } => false,
    HELoweredInstr::NegateInplace { op1: _ } => true,
    HELoweredInstr::Mul { id: _, op1: _, op2: _ } => false,
    HELoweredInstr::MulInplace { op1: _, op2: _ } => true,
    HELoweredInstr::MulPlain { id: _, op1: _, op2: _ } => false,
    HELoweredInstr::MulPlainInplace { op1: _, op2: _ } => true,
    HELoweredInstr::Rot { id: _, op1: _, op2: _ } => false,
    HELoweredInstr::RotInplace { op1: _, op2: _ } => true,
    HELoweredInstr::RelinearizeInplace { op1: _ } => true
});

handlebars_helper!(instr_name: |instr: HELoweredInstr| match instr {
    HELoweredInstr::Add { id: _, op1: _, op2: _} => "add",
    HELoweredInstr::AddInplace { op1: _, op2: _} => "add_inplace",
    HELoweredInstr::AddPlain { id: _, op1: _, op2: _ } => "add_plain",
    HELoweredInstr::AddPlainInplace { op1: _, op2: _ } => "add_plain_inplace",
    HELoweredInstr::Sub { id: _, op1: _, op2: _} => "sub",
    HELoweredInstr::SubInplace { op1: _, op2: _} => "sub_inplace",
    HELoweredInstr::SubPlain { id: _, op1: _, op2: _ } => "sub_plain",
    HELoweredInstr::SubPlainInplace { op1: _, op2: _ } => "sub_plain_inplace",
    HELoweredInstr::Negate { id: _, op1: _ } => "negate",
    HELoweredInstr::NegateInplace { op1: _ } => "negate_inplace",
    HELoweredInstr::Mul { id: _, op1: _, op2: _ } => "multiply",
    HELoweredInstr::MulInplace { op1: _, op2: _ } => "multiply_inplace",
    HELoweredInstr::MulPlain { id: _, op1: _, op2: _ } => "multiply_plain",
    HELoweredInstr::MulPlainInplace { op1: _, op2: _ } => "multiply_plain_inplace",
    HELoweredInstr::Rot { id: _, op1: _, op2: _ } => "rotate_rows",
    HELoweredInstr::RotInplace { op1: _, op2: _ } => "rotate_rows_inplace",
    HELoweredInstr::RelinearizeInplace { op1: _ } => "relinearize_inplace"
});

pub struct CodeGenerator<'a> {
    handlebars: Handlebars<'a>
}

impl CodeGenerator<'_> {
    pub fn new(template_file: &str) -> Self {
        let template_str =
            std::fs::read_to_string(template_file)
            .expect(&format!("Could not read file {}", template_file));

        let mut codegen = CodeGenerator { handlebars: Handlebars::new() };
        codegen.handlebars.register_helper("instr_is_inplace", Box::new(instr_is_inplace));
        codegen.handlebars.register_helper("instr_is_binary", Box::new(instr_is_binary));
        codegen.handlebars.register_helper("instr_name", Box::new(instr_name));
        codegen.handlebars.register_template_string("template", template_str)
            .expect("Could not register template");

        codegen
    }

    pub fn render_to_file(&self, lowered_prog: &HELoweredProgram, filename: &str) -> Result<(), RenderError> {
        let f = File::create(filename)?;
        self.handlebars.render_to_write("template", lowered_prog, f)
    }

    pub fn render_to_str(&self, lowered_prog: &HELoweredProgram) -> Result<String, RenderError> {
        self.handlebars.render("template", lowered_prog)
    }
}