#[macro_use] extern crate lalrpop_util;

/// main.rs
/// Vectorizer for homomorphic encryption circuits

use clap::Parser;
use handlebars::{Handlebars, handlebars_helper};
use egg::RecExpr;
use log::*;
use std::{collections::HashMap, fs::File};

use he_vectorizer::circ::{
    lowering::{
        program::HEProgram,
        lowered_program::{HELoweredInstr, HELoweredProgram}, code_gen::CodeGenerator,
    },
    optimizer::{HEOptimizerCircuit, ExtractorType, optimize}};

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

    /// don't run equality saturation to optimize program
    #[clap(short = 'p', long = "passthru")]
    passthrough: bool,

    /// don't inline instructions
    #[clap(short = 'n', long = "noinline")]
    noinline: bool,
}

fn main() {
    env_logger::init();

    let args = Arguments::parse();
    let input_str =
        std::fs::read_to_string(&args.file)
        .expect(&format!("Could not read file {}", &args.file));

    // parse the expression, the type annotation tells it which Language to use
    let init_expr: RecExpr<HEOptimizerCircuit> = input_str.parse().unwrap();
    let init_prog = HEProgram::from(&init_expr);
    // info!("Initial HE expr:\n{}", init_expr.pretty(80));
    info!("Initial HE program (muldepth {}, latency {}ms):",
        init_prog.get_muldepth(),
        init_prog.get_latency()
    );

    let opt_expr =
        if !args.passthrough {
            optimize(&init_expr, args.size, args.duration, args.extractor)

        } else {
            init_expr
        };

    let opt_prog = HEProgram::from(&opt_expr);

    if !args.passthrough {
        info!("Optimized HE program (muldepth {}, latency {}ms):",
            opt_prog.get_muldepth(),
            opt_prog.get_latency()
        );
    }

    let lowered_prog = HELoweredProgram::lower_program(&opt_prog, &HashMap::new(), args.size, args.noinline);
    let codegen = CodeGenerator::new(&args.template);

    if args.outfile.len() > 1 {
        codegen.render_to_file(&lowered_prog, &args.outfile).unwrap();
        info!("Wrote program to {}", &args.outfile);


    } else {
        // info!("{}", handlebars.render("template", &lower_program(&init_prog, args.size)).unwrap());
        // info!("Optimized HE expr:\n{}", opt_expr.pretty(80));

        let output = codegen.render_to_str(&lowered_prog).unwrap();
        info!("{}", output);
    }

    // let vec_size = 16;
    // let sym_store: HESymStore = init_prog.gen_sym_store(vec_size, -10..=10);
    // let init_out = interp_program(&sym_store, &init_prog, vec_size);
    // let opt_out = interp_program(&sym_store, &opt_prog, vec_size);

    // // values of the last instructions should be equal
    // info!("output for init prog: {}", init_out.unwrap());
    // info!("output for opt prog: {}", opt_out.unwrap());
}
