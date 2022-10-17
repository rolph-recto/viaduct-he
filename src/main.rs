extern crate lalrpop_util;

/// main.rs
/// Vectorizer for homomorphic encryption circuits

use clap::Parser;
use egg::RecExpr;
use log::*;
use std::collections::HashMap;

use he_vectorizer::circ::{
    lowering::{
        program::HEProgram,
        lowered_program::HELoweredProgram, code_gen::CodeGenerator,
    },
    optimizer::{HEOptCircuit, ExtractorType, Optimizer, HELatencyModel}, HECircuitStore, circ_gen::ClientTransform};

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

    /// don't inline instructions
    #[clap(short = 'n', long = "noinplace")]
    noinplace: bool,
}

fn main() {
    env_logger::init();

    let args = Arguments::parse();
    let input_str =
        std::fs::read_to_string(&args.file)
        .expect(&format!("Could not read file {}", &args.file));

    let latency_model = HELatencyModel::default();

    // parse the expression, the type annotation tells it which Language to use
    let init_expr: RecExpr<HEOptCircuit> = input_str.parse().unwrap();
    let init_prog = HEProgram::from(&init_expr);
    // info!("Initial HE expr:\n{}", init_expr.pretty(80));
    info!("Initial HE program (muldepth {}, latency {}ms):",
        init_prog.get_muldepth(),
        init_prog.get_latency(&latency_model)
    );

    let opt_expr =
        if args.duration > 0 {
            let optimizer = Optimizer::new(args.size);
            optimizer.optimize(&init_expr, args.size, args.duration, args.extractor)

        } else {
            init_expr
        };

    let opt_prog = HEProgram::from(&opt_expr);

    if args.duration > 0 {
        info!("Optimized HE program (muldepth {}, latency {}ms):",
            opt_prog.get_muldepth(),
            opt_prog.get_latency(&latency_model)
        );
    }

    let lowered_prog =
        HELoweredProgram::lower_program(
            args.size,
            args.noinplace,
            &opt_prog, 
            &HECircuitStore::default(),
            HashMap::from([
                (String::from("cimg"),
                ClientTransform::Pad(
                    Box::new(ClientTransform::InputArray(String::from("img"))),
                    im::vector![(2,2),(2,2)]
                ))
            ]),
            HashMap::new(),
            HashMap::new());
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
