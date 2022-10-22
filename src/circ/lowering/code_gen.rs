use std::fs::File;

use handlebars::{Handlebars, handlebars_helper, RenderError};

use super::lowered_program::{HELoweredInstr, HELoweredProgram};

handlebars_helper!(instr_is_binary: |instr: HELoweredInstr| instr.is_binary());

handlebars_helper!(instr_is_inplace: |instr: HELoweredInstr| instr.is_inplace());

handlebars_helper!(instr_name: |instr: HELoweredInstr| instr.name());

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