use either::{Either,Left,Right};

use std::map::{Map,HashMap,Set};

use hashlife::{rank_to_size,size_to_rank,split,case_results};

struct Board {
  live_cells: Set<(uint,uint)>,
  rows: uint,
  cols: uint
}

// Converting cells to boards
fn cell_to_board(rank: uint, c: Cell) -> Board {
  let b = Board {
    live_cells: HashMap(),
    rows: rank_to_size(rank),
    cols: rank_to_size(rank)
  };
  add_cell_to_board(&b, rank, c, 0, 0);
  b
}

fn add_cell_to_board(b: &Board, rank: uint, c: Cell, xoff: uint, yoff: uint) {
  match case_results(rank, &split(c)) {
    Right(bits) => {
      // Add in the bits
      for uint::range(0, 16) |i| {
        if (bits >> (15-i)) & 1 == 1 {
          b.live_cells.insert((xoff + (i%4), yoff + (i/4)), ());
        }
      }
    }
    Left(quads) => {
      // Recurse
      let off = rank_to_size(rank-1);
      for quads.eachi |i,cell| {
        add_cell_to_board(b, rank-1, *cell, xoff + (i%2)*off, yoff + (i/2)*off);
      }
    }
  }
}


// Converting boards to cells

// Roughly centers the board in a new cell of the smallest rank capable of
// containing it.
fn board_to_cell(world: &World, board: &Board) -> (uint,Cell) {
  let rank = size_to_rank(uint::max(board.rows, board.cols));
  let size = rank_to_size(rank);

  let xoff = (size - board.cols) / 2,
      yoff = (size - board.rows) / 2;
  (rank,
   do cell_from_fn(world, rank, 0, 0) |x,y| {
     board.live_cells.contains_key((x-xoff,y-yoff))
   })
}

fn cell_from_fn
  (w: &World, rank: uint, xoff: uint, yoff: uint, state: fn&(uint,uint) -> bool)
  -> Cell
{
  if (rank > 0) {
    let off = rank_to_size(rank-1);
    let f: fn&(uint,uint) -> Result = |i,j| {
      Cell(cell_from_fn(w, rank-1, xoff+i, yoff+j, state))
    };
    w.merge(rank, [f(0,0),   f(off,0),
                   f(0,off), f(off,off)])
  } else {
    let mut bits: u16 = 0;
    for uint::range(0,16) |i| {
      let bit = if state(xoff + (i % 4), yoff + (i / 4)) { 1 } else { 0 };
      bits = (bits << 1) | bit;
    }
    w.makeFour(bits)
  }
}
