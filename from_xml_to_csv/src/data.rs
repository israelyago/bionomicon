#[derive(Debug)]
pub struct BiologicalData {
    // accession: [u8; 10], // To save some bytes in memory, may not be needed
    accession: String,
    sequence: String,
    is_enzyme: bool,
}

impl BiologicalData {
    pub fn new(accession: String, sequence: String, is_enzyme: bool) -> Self {
        // let bytes = accession.into_bytes();
        // let slice = bytes.as_slice();
        // if slice.len() == 10 {
        //     let array: [u8; 10] = [
        //         slice[0],
        //         slice[1],
        //         slice[2],
        //         slice[3],
        //         slice[4],
        //         slice[5],
        //         slice[6],
        //         slice[7],
        //         slice[8],
        //         slice[9],
        //     ];
        //     Self {
        //         accession: array,
        //         sequence,
        //     }
        // } else {
        //     let array: [u8; 10] = [
        //         slice[0],
        //         slice[1],
        //         slice[2],
        //         slice[3],
        //         slice[4],
        //         slice[5],
        //         0,
        //         0,
        //         0,
        //         0,
        //     ];
        //     Self {
        //         accession: array,
        //         sequence,
        //     }
        // }
        Self {
            accession,
            sequence,
            is_enzyme,
        }
    }

    pub fn accession(&self) -> &str {
        // std::str::from_utf8(self.accession.iter().as_slice()).unwrap()
        self.accession.as_ref()
    }

    pub fn sequence(&self) -> &str {
        self.sequence.as_ref()
    }

    pub fn is_enzyme(&self) -> bool {
        self.is_enzyme
    }
}