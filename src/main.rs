extern crate lalrpop_util;

use std::{fs::File, io::Write, time::{Duration, Instant}};

/// main.rs
/// Vectorizer for homomorphic encryption circuits
use clap::Parser;
use log::*;

use he_vectorizer::{
    circ::{
        optimizer::ExtractorType,
        materializer::{DefaultArrayMaterializer, Materializer, InputArrayMaterializer},
        partial_eval::HEPartialEvaluator
    },
    lang::{
        index_elim::IndexElimination,
        parser::ProgramParser,
        elaborated::Elaborator
    },
    scheduling::Schedule,
    program::{
        lowering::CircuitLowering,
        backend::{
            HEBackend,
            pyseal::SEALBackend
        }
    },
};

#[derive(Parser)]
#[clap(author, version, about = "optimizer for for vectorized homomorphic encryption circuits", long_about = None)]
struct HEArguments {
    /// file to parse as input
    #[clap(value_parser)]
    file: String,

    /// template file for output program
    #[clap(
        short = 't',
        long = "template",
        value_parser,
    )]
    template: Option<String>,

    /// file for output program
    #[clap(
        short = 'o',
        long = "outfile",
        value_parser,
    )]
    outfile: Option<String>,

    /// duration in seconds to run optimizer until timeout (if 0, duration is unbounded)
    #[clap(short = 'd', long = "duration", value_parser, default_value_t = 20)]
    duration: usize,

    /// duration in seconds to run equality saturation until timeout
    #[clap(short = 'e', long = "extractor", value_enum, default_value_t = ExtractorType::Greedy)]
    extractor: ExtractorType,

    /// vector size
    #[clap(short = 's', long = "size", value_parser, default_value_t = 2048)]
    size: usize,

    /// don't inline instructions
    #[clap(short = 'n', long = "noinplace")]
    noinplace: bool,
}

// fn dumpinfo() {}

fn main() {
    env_logger::init();

    let args = HEArguments::parse();
    let input_str =
        std::fs::read_to_string(&args.file).expect(&format!("Could not read file {}", &args.file));

    info!("parsing...");
    let source = ProgramParser::new().parse(&input_str).unwrap();

    info!("elaboration...");
    let elaborated = Elaborator::new().run(source);

    let inline_set = elaborated.get_default_inline_set();
    let array_group_map = elaborated.get_default_array_group_map();

    info!("index elimination...");
    let res_index_elim = IndexElimination::new().run(&inline_set, &array_group_map, elaborated);

    let inlined = res_index_elim.unwrap();
    let init_schedule = Schedule::gen_initial_schedule(&inlined);

    info!("materialization...");
    let array_materializers: Vec<Box<dyn InputArrayMaterializer>> = 
        vec![Box::new(DefaultArrayMaterializer::new())];
    let materializer = Materializer::new(array_materializers);

    let res_materialize =
        materializer.run(&inlined, &init_schedule);

    let circuit = res_materialize.unwrap();

    // TODO add optimizer

    info!("partial evaluation...");
    let circuit_pe = HEPartialEvaluator::new().run(circuit);

    info!("circuit lowering...");
    let program = CircuitLowering::new().run(circuit_pe);

    info!("compiling with PySEAL backend...");
    let seal_backend =
        SEALBackend::new(
            args.template.clone(), 
            !args.noinplace,
            args.size
        );

    let mut code_str: String = String::new();
    seal_backend.compile(program, &mut code_str).unwrap();

    match args.outfile {
        Some(outfile) => {
            let mut file = File::create(&outfile).unwrap();
            file.write(code_str.as_bytes()).unwrap();
            info!("wrote generated code to {}", outfile);
        },

        None => {
            info!("printed generated code to stdout");
            println!("{}", code_str)
        }
    }
}
