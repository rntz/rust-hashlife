use std::map::HashMap;
use dvec::DVec;

use hashlife::World;

// Main program
fn main() {
  let _world = World {
    cache4x4: HashMap(),
    caches: DVec(),
  };

  io::println("stix");
}
