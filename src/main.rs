#[macro_use] extern crate lalrpop_util;

mod lang;

/// main.rs
/// Vectorizer for homomorphic encryption circuits

use clap::Parser;
use handlebars::{Handlebars, handlebars_helper};
use egg::RecExpr;
use log::*;
use std::fs::File;

use he_vectorizer::ir::{instr::{gen_program, HEProgram}, expr::HEExpr, lowered::HELoweredInstr};
use he_vectorizer::ir::{lowered::lower_program, optimizer::{ExtractorType, optimize}};

#[derive(Parser)]
#[clap(author, version, about = "optimizer for for vectorized homomorphic encryption circuits", long_about = None)]
struct Arguments {
    /// file to parse as input
    #[clap(value_parser)]
    file: String,

    /// template file for output program
    #[clap(short = 't', long = "template", value_parser, default_value = "template.txt")]
    template: String,

    /// file for output program
    #[clap(short = 'o', long = "outfile", value_parser, default_value = "")]
    outfile: String,

    /// duration in seconds to run optimizer until timeout (if 0, duration is unbounded)
    #[clap(short = 'd', long = "duration", value_parser, default_value_t = 20)]
    duration: usize,

    /// duration in seconds to run equality saturation until timeout
    #[clap(short = 'e', long = "extractor", value_enum, default_value_t = ExtractorType::GREEDY)]
    extractor: ExtractorType,

    /// vector size
    #[clap(short = 's', long = "size", value_parser, default_value_t = 8192)]
    size: usize,

    /// let input pass through and don't optimize it
    #[clap(short = 'p', long = "passthru")]
    passthrough: bool,
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
    HELoweredInstr::RotInplace { op1: _, op2: _ } => true,
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
    HELoweredInstr::RotInplace { op1: _, op2: _ } => true,
    HELoweredInstr::RelinearizeInplace { op1: _ } => true
});

handlebars_helper!(instr_name: |instr: HELoweredInstr| match instr {
    HELoweredInstr::Add { id: _, op1: _, op2: _} => "add",
    HELoweredInstr::AddInplace { op1: _, op2: _} => "add_inplace",
    HELoweredInstr::AddPlain { id: _, op1: _, op2: _ } => "add_plain",
    HELoweredInstr::AddPlainInplace { op1: _, op2: _ } => "add_plain_inplace",
    HELoweredInstr::Mul { id: _, op1: _, op2: _ } => "multiply",
    HELoweredInstr::MulInplace { op1: _, op2: _ } => "multiply_inplace",
    HELoweredInstr::MulPlain { id: _, op1: _, op2: _ } => "multiply_plain",
    HELoweredInstr::MulPlainInplace { op1: _, op2: _ } => "multiply_plain_inplace",
    HELoweredInstr::Rot { id: _, op1: _, op2: _ } => "rotate_rows",
    HELoweredInstr::RotInplace { op1: _, op2: _ } => "rotate_rows_inplace",
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
    let init_expr: RecExpr<HEExpr> = input_str.parse().unwrap();
    let init_prog = gen_program(&init_expr);
    // info!("Initial HE expr:\n{}", init_expr.pretty(80));
    info!("Initial HE program (muldepth {}, latency {}ms):",
        init_prog.get_muldepth(),
        init_prog.get_latency()
    );

    let mut handlebars = Handlebars::new();
    handlebars.register_helper("instr_is_inplace", Box::new(instr_is_inplace));
    handlebars.register_helper("instr_is_binary", Box::new(instr_is_binary));
    handlebars.register_helper("instr_name", Box::new(instr_name));
    handlebars.register_template_string("template", template_str)
        .expect("Could not register template");

    let opt_expr =
        if !args.passthrough {
            optimize(&init_expr, args.size as i32, args.duration, args.extractor)

        } else {
            init_expr.clone()
        };

    let opt_prog: HEProgram = gen_program(&opt_expr);

    if !args.passthrough {
        info!("Optimized HE program (muldepth {}, latency {}ms):",
            opt_prog.get_muldepth(),
            opt_prog.get_latency()
        );
    }

    let lowered_prog = lower_program(&opt_prog, args.size);

    if args.outfile.len() > 1 {
        let f =  File::create(&args.outfile).unwrap();
        handlebars.render_to_write("template", &lowered_prog, f).unwrap();
        info!("Wrote program to {}", &args.outfile);

    } else {
        // info!("{}", handlebars.render("template", &lower_program(&init_prog, args.size)).unwrap());
        // info!("Optimized HE expr:\n{}", opt_expr.pretty(80));

        info!("{}", handlebars.render("template", &lowered_prog).unwrap());
    }

    // let vec_size = 16;
    // let sym_store: HESymStore = init_prog.gen_sym_store(vec_size, -10..=10);
    // let init_out = interp_program(&sym_store, &init_prog, vec_size);
    // let opt_out = interp_program(&sym_store, &opt_prog, vec_size);

    // // values of the last instructions should be equal
    // info!("output for init prog: {}", init_out.unwrap());
    // info!("output for opt prog: {}", opt_out.unwrap());
}
