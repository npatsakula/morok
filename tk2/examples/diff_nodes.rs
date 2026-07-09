// Quick diff: find the first differing node between pipe and pipe2 at the single-warp shape.
use svod_tk2::{TileId, matmul_lds_kblock_mw_pipe, matmul_lds_kblock_mw_pipe2};
fn main() {
    let pipe = matmul_lds_kblock_mw_pipe(1024, 1024, 1024, 64, 64, 1, 1, 64);
    let pipe2 = matmul_lds_kblock_mw_pipe2(1024, 1024, 1024, 64, 64, 1, 1, 64);
    let len = pipe.ir.len().min(pipe2.ir.len());
    for id in 0..len {
        let id = TileId(id as u32);
        let np = pipe.ir.node(id);
        let np2 = pipe2.ir.node(id);
        if np != np2 {
            println!("FIRST DIFF at node {id:?}:");
            println!("  pipe:  {np:?}");
            println!("  pipe2: {np2:?}");
            // print a few nodes before for context
            if id.0 > 0 {
                let prev = TileId(id.0 - 1);
                println!("  (prev pipe:  {:?})", pipe.ir.node(prev));
                println!("  (prev pipe2: {:?})", pipe2.ir.node(prev));
            }
            break;
        }
    }
    println!("pipe len={} pipe2 len={}", pipe.ir.len(), pipe2.ir.len());
    // if all common nodes match, the extra node is at the end
    if pipe.ir.len() != pipe2.ir.len() {
        let extra = if pipe2.ir.len() > pipe.ir.len() { &pipe2 } else { &pipe };
        let id = TileId(len as u32);
        println!("EXTRA node at {id:?}: {:?}", extra.ir.node(id));
    }
}
