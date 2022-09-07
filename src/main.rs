/// main.rs
/// Vectorizer for homomorphic encryption circuits

use clap::Parser;
use handlebars::{Handlebars, handlebars_helper};
use egg::RecExpr;
use lang::{instr::{gen_program, HEProgram}, expr::HE, lowered::HELoweredInstr};
use log::*;
use std::fs::File;

use crate::{lang::lowered::lower_program, optimizer::{ExtractorType, optimize}};

mod lang;
mod optimizer;

#[derive(Parser)]
#[clap(author, version, about = "optimizer for for vectorized homomorphic encryption circuits", long_about = None)]
struct Arguments {
    /// file to parse as input
    #[clap(value_parser)]
    file: String,

    /// template file for output program
    #[clap(short = 'p', long = "template", value_parser, default_value = "template.txt")]
    template: String,

    /// file for output program
    #[clap(short = 'o', long = "outfile", value_parser, default_value = "")]
    outfile: String,

    /// duration in seconds to run equality saturation until timeout
    #[clap(short = 'd', long = "duration", value_parser, default_value_t = 20)]
    duration: u64,

    /// duration in seconds to run equality saturation until timeout
    #[clap(short = 'e', long = "extractor", value_enum, default_value_t = ExtractorType::GREEDY)]
    extractor: ExtractorType,
}

handlebars_helper!(instr_is_binary: |instr: HELoweredInstr| match instr {
    HELoweredInstr::Add { id: _, op1: _, op2: _} => true,
    HELoweredInstr::AddInplace { op1: _, op2: _} => true,
    HELoweredInstr::AddPlain { id: _, op1: _, op2: _ } => true,
    HELoweredInstr::AddPlainInplace { op1: _, op2: _ } => true,
    HELoweredInstr::Mul { id: _, op1: _, op2: _ } => true,
    HELoweredInstr::MulInplace { op1: _, op2: _ } => true,
    HELoweredInstr::MulPlain { id: _, op1: _, op2: _ } => true,
    HELoweredInstr::MulPlainInplace { op1: _, op2: _ } => true,
    HELoweredInstr::Rot { id: _, op1: _, op2: _ } => true,
    HELoweredInstr::RelinearizeInplace { op1: _ } => false
});

handlebars_helper!(instr_is_inplace: |instr: HELoweredInstr| match instr {
    HELoweredInstr::Add { id: _, op1: _, op2: _} => false,
    HELoweredInstr::AddInplace { op1: _, op2: _} => true,
    HELoweredInstr::AddPlain { id: _, op1: _, op2: _ } => false,
    HELoweredInstr::AddPlainInplace { op1: _, op2: _ } => true,
    HELoweredInstr::Mul { id: _, op1: _, op2: _ } => false,
    HELoweredInstr::MulInplace { op1: _, op2: _ } => true,
    HELoweredInstr::MulPlain { id: _, op1: _, op2: _ } => false,
    HELoweredInstr::MulPlainInplace { op1: _, op2: _ } => true,
    HELoweredInstr::Rot { id: _, op1: _, op2: _ } => false,
    HELoweredInstr::RelinearizeInplace { op1: _ } => true
});

handlebars_helper!(instr_name: |instr: HELoweredInstr| match instr {
    HELoweredInstr::Add { id: _, op1: _, op2: _} => "add",
    HELoweredInstr::AddInplace { op1: _, op2: _} => "add_inplace",
    HELoweredInstr::AddPlain { id: _, op1: _, op2: _ } => "add_plain",
    HELoweredInstr::AddPlainInplace { op1: _, op2: _ } => "add_plain_inplace",
    HELoweredInstr::Mul { id: _, op1: _, op2: _ } => "mul",
    HELoweredInstr::MulInplace { op1: _, op2: _ } => "mul_inplace",
    HELoweredInstr::MulPlain { id: _, op1: _, op2: _ } => "mul_plain",
    HELoweredInstr::MulPlainInplace { op1: _, op2: _ } => "mul_plain_inplace",
    HELoweredInstr::Rot { id: _, op1: _, op2: _ } => "rotate",
    HELoweredInstr::RelinearizeInplace { op1: _ } => "relinearize_inplace"
});

fn main() {
    env_logger::init();

    let args = Arguments::parse();
    let input_str =
        std::fs::read_to_string(&args.file)
        .expect(&format!("Could not read file {}", &args.file));

    let template_str =
        std::fs::read_to_string(&args.template)
        .expect(&format!("Could not read file {}", &args.template));


    // parse the expression, the type annotation tells it which Language to use
    let init_expr: RecExpr<HE> = input_str.parse().unwrap();
    let init_prog = gen_program(&init_expr);

    let opt_expr = optimize(&init_expr, args.duration, args.extractor);
    let opt_prog: HEProgram = gen_program(&opt_expr);

    let mut handlebars = Handlebars::new();
    handlebars.register_helper("instr_is_inplace", Box::new(instr_is_inplace));
    handlebars.register_helper("instr_is_binary", Box::new(instr_is_binary));
    handlebars.register_helper("instr_name", Box::new(instr_name));
    handlebars.register_template_string("t", template_str)
        .expect("Could not register template");

    if args.outfile.len() > 1 {
        let f =  File::create(&args.outfile).unwrap();
        handlebars.render_to_write("t", &lower_program(&opt_prog), f).unwrap();
        info!("Initial HE program (muldepth {}, latency {}ms):\n",
            init_prog.get_muldepth(),
            init_prog.get_latency()
        );
        info!("Optimized HE program (muldepth {}, latency {}ms):\n",
            opt_prog.get_muldepth(),
            opt_prog.get_latency()
        );
        info!("Generated optimized program to {}", &args.outfile);

    } else {
        info!("Initial HE expr:\n{}", init_expr.pretty(80));
        info!("Initial HE program (muldepth {}, latency {}ms):\n{}",
            init_prog.get_muldepth(),
            init_prog.get_latency(),
            handlebars.render("t", &lower_program(&init_prog)).unwrap()
        );

        info!("Optimized HE expr:\n{}", opt_expr.pretty(80));
        info!("Optimized HE program (muldepth {}, latency {}ms):\n{}",
            opt_prog.get_muldepth(),
            opt_prog.get_latency(),
            handlebars.render("t", &lower_program(&opt_prog)).unwrap()
        );
    }

    // let vec_size = 16;
    // let sym_store: HESymStore = init_prog.gen_sym_store(vec_size, -10..=10);
    // let init_out = interp_program(&sym_store, &init_prog, vec_size);
    // let opt_out = interp_program(&sym_store, &opt_prog, vec_size);

    // // values of the last instructions should be equal
    // info!("output for init prog: {}", init_out.unwrap());
    // info!("output for opt prog: {}", opt_out.unwrap());
}
