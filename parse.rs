use io::{Reader, ReaderUtil, Writer, WriterUtil};

use std::map::{HashMap};

use board::Board;

// Parsing plaintext into boards.
//
// Parses the pattern into the smallest cell that will contain it, roughly
// centered in that cell.
fn parse_plaintext<T:Reader>(input: &T) -> Board {
  // for now, we don't support comments or headers, sorry.
  //plaintext_drop_leading_comments(input);

  let set = HashMap();
  let mut i = 0u,j = 0u, cols = 0u;

  while !input.eof() {
    // shouldn't have to deref here, but autoderef is broken
    match (*input).read_char() {
      '\n' => {
        if cols < j { cols = j }
        i += 1;
        j = 0;
      }
      'O' => { set.insert((i,j), ()); j += 1; }
      _ => { j += 1; }
    }
  }

  Board {
    live_cells: set,
    rows: i,
    cols: cols
  }
}

