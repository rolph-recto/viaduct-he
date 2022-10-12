use he_vectorizer::circ::{*, circ_gen::HECircuitGenerator};
use he_vectorizer::lang::{
    parser::ProgramParser,
    normalized::Normalizer,
    source::SourceProgram,
    typechecker::TypeChecker,
};


#[test]
fn test_parse_positive() {
    let parser = ProgramParser::new();
    assert!(parser.parse("42").is_ok());
    assert!(parser.parse("(42)").is_ok());
    assert!(parser.parse("42 + 56").is_ok());
    assert!(parser.parse("42 * 56").is_ok());
    assert!(parser.parse("for x: (0,16) { 42 }").is_ok());
    assert!(parser.parse("for x: (0,16) { for y: (0, 16) { img[x][y] + 2  }}").is_ok());
    assert!(parser.parse("
        let img2 = for x: (0,16) { for y: (0, 16) { img[x][y] + 2  }}
        img2 + img2
    ").is_ok());
}

#[test]
fn test_parse_negative() {
    let parser = ProgramParser::new();
    assert!(parser.parse("for x: (0,16) in { 42").is_err());
}

#[test]
fn test_typechecker_positive() {
    let parser = ProgramParser::new();
    let typechecker = TypeChecker::new();

    let prog1 = parser.parse("42").unwrap();
    let prog2 = parser.parse("42 + 56").unwrap();
    let prog3 = parser.parse("
        input img: [(0,16),(0,16)]
        for x: (0,16) { img[x] }
    ").unwrap();

    assert!(typechecker.run(&prog1).is_ok());
    assert!(typechecker.run(&prog2).is_ok());
    assert!(typechecker.run(&prog3).is_ok());
}

#[test]
fn test_typechecker_negative() {
    let parser = ProgramParser::new();
    let typechecker = TypeChecker::new();

    let prog1 = parser.parse("sum(42)").unwrap();
    let prog2 = parser.parse("
        input img: [(0,16),(0,16)]
        for x: (0,16) {
            for y: (0, 16) {
                for z: (0, 16) { img[x][y][z] }
            }
        }
    ").unwrap();
    let prog3 = parser.parse("
        input img: [(0,16),(0,16)]
        let next = img + img
        for x: (0,16) {
            for y: (0, 16) {
                img[x][y] + next
            }
        }
    ").unwrap();

    assert!(typechecker.run(&prog1).is_err());
    assert!(typechecker.run(&prog2).is_err());
    assert!(typechecker.run(&prog3).is_err());
}

#[test]
fn imgblur() {
    let parser = ProgramParser::new();
    let program: SourceProgram =
        parser.parse("
            input img: [(0,16),(0,16)]
            for x: (0, 16) {
                for y: (0, 16) {
                    img[x-1][y-1] + img[x+1][y+1]
                }
            }
        ").unwrap();

    let res = Normalizer::new().run(&program);
    assert!(res.is_ok());
    println!("{}", res.unwrap().expr)
}

#[test]
fn test_mask_iteration_domain() {
    let shape: im::Vector<usize> = im::vector![4, 2, 3];
    let iteration_dom = HECircuitGenerator::get_iteration_domain(&shape);
    assert_eq!(iteration_dom.len(), 24);
    assert_eq!(iteration_dom[0], im::vector![0,0,0]);
    assert_eq!(iteration_dom.last().unwrap().clone(), im::vector![3,1,2]);
}

#[test]
fn matmatmul() {
    let parser = ProgramParser::new();
    let prog: SourceProgram =
        parser.parse("
        input A: [(0,4),(0,4)]
        input B: [(0,4),(0,4)]
        let x = A + B
        for i: (0,4) {
            for j: (0,4) {
                sum(for k: (0,4) { A[i][k] * B[k][j] })
            }
        }
        ").unwrap();

    let res = Normalizer::new().run(&prog);
    assert!(res.is_ok());
    println!("{}", res.unwrap().expr);
}