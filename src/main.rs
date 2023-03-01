extern crate lalrpop_util;

/// main.rs
/// Vectorizer for homomorphic encryption circuits

use clap::Parser;
use log::*;

use he_vectorizer::{
    lang::{parser::ProgramParser, index_elim::IndexElimination},
    circ::optimizer::ExtractorType,
};

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
    #[clap(short = 'e', long = "extractor", value_enum, default_value_t = ExtractorType::Greedy)]
    extractor: ExtractorType,

    /// vector size
    #[clap(short = 's', long = "size", value_parser, default_value_t = 8192)]
    size: usize,

    /// don't inline instructions
    #[clap(short = 'n', long = "noinplace")]
    noinplace: bool,
}

fn dumpinfo() {

}

fn main() {
    env_logger::init();

    let args = Arguments::parse();
    let input_str =
        std::fs::read_to_string(&args.file)
        .expect(&format!("Could not read file {}", &args.file));

    let parser = ProgramParser::new();
    let src_program = parser.parse(&input_str).unwrap();

    /*
    let index_elim = IndexElimination::new();
    let indfree_program = index_elim.run(&src_program).unwrap();

    let circ_gen = HECircuitGenerator::new();
    let (circ, store) = circ_gen.gen_circuit(&indfree_program).unwrap();

    let latency_model = HELatencyModel::default();

    // parse the expression, the type annotation tells it which Language to use
    // let init_circ: RecExpr<HEOptCircuit> = input_str.parse().unwrap();

    let init_circ = circ.to_opt_circuit();
    let init_prog = HEProgram::from(&init_circ);

    info!("Initial HE program (muldepth {}, latency {}ms):",
        init_prog.get_muldepth(),
        init_prog.get_latency(&latency_model)
    );

    let opt_circ =
        if args.duration > 0 {
            let optimizer = Optimizer::new(args.size);
            optimizer.optimize(&init_circ, args.size, args.duration, args.extractor)

        } else {
            init_circ
        };

    let opt_prog = HEProgram::from(&opt_circ);

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
            &store,
            indfree_program.client_store,
            HashMap::new(),
            HashMap::new()
        );
    let codegen = CodeGenerator::new(&args.template);

    if args.outfile.len() > 1 {
        codegen.render_to_file(&lowered_prog, &args.outfile).unwrap();
        info!("Wrote program to {}", &args.outfile);

    } else {
        let output = codegen.render_to_str(&lowered_prog).unwrap();
        info!("{}", output);
    }
    */
}
