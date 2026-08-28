struct Grid {
    width: usize,
    height: usize,
    current: Vec<u8>,
    next: Vec<u8>,
}

impl Grid {
    // 1. Initialize the grid
    fn new(width: usize, height: usize) -> Self {
        let total_size = width * height;
        
        Grid {
            width,
            height,
            // vec![0; size] creates an array of zeros of that exact size
            current: vec![0; total_size],
            next: vec![0; total_size],
        }
    }

    // 2. The 2D to 1D translation formula
    fn get_index(&self, x: usize, y: usize) -> usize {
        y * self.width + x
    }
}

fn main() {
    let grid = Grid::new(5, 5);
    println!("Grid initialized with {} cells!", grid.current.len());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_index() {
        let grid = Grid::new(5, 5);
        
        // Top-left corner
        assert_eq!(grid.get_index(0, 0), 0);
        
        // Start of the second row (skipped one full row of 5)
        assert_eq!(grid.get_index(0, 1), 5);
        
        // Middle of the grid (Row 2, Column 3) -> (2 * 5) + 3 = 13
        assert_eq!(grid.get_index(3, 2), 13);
        
        // Bottom-right corner (Row 4, Column 4) -> (4 * 5) + 4 = 24
        assert_eq!(grid.get_index(4, 4), 24);
    }
}
