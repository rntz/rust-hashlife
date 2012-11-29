use dvec::DVec;
use io::{WriterUtil};           // for write_char

use std::map::HashMap;

use hashlife::{Cell,Two};
use board::{Board,cell_to_board,board_to_cell};
use parse::{parse_plaintext,dump_plaintext};

// Main program
fn main() {
  // Read a board.
  warn!("parsing");
  let stdin = io::stdin(), stdout = io::stdout();
  let board = parse_plaintext(&stdin);
  dump_plaintext(&stdout, &board);

  // Create a world.
  warn!("creating world");
  let world = World {
    cache4x4: HashMap(),
    caches: DVec(),
  };

  // Convert board to cell.
  warn!("converting board to cell");
  //log(warn, board);
  let mut (rank,cell) = board_to_cell(&world, &board);

  // Simulate, printing out result at each step.
  loop {
    warn!("dumping");
    dump_plaintext(&stdout, &cell_to_board(rank, cell));
    stdout.write_char('\n');

    // Compute future.
    warn!("calculating result");
    cell = match world.result(rank, cell) {
      Cell(c) => c,
      Two(_) => { break }
    };
    rank -= 1;
  }
}
