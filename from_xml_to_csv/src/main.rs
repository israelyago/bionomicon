use extractor::data::BiologicalData;
use extractor::parser::extract_entry;
use quick_xml::events::Event;
use quick_xml::reader::Reader;
use std::ffi::OsStr;
use std::fs::File;
use std::path::PathBuf;
use clap::Parser;
use std::time::Instant;

#[derive(Debug)]
enum AppError {
    Xml(quick_xml::Error),
    InvalidInputPath,
    IOError(std::io::Error),
    CSVError(csv::Error),
}

impl From<quick_xml::Error> for AppError {
    fn from(error: quick_xml::Error) -> Self {
        Self::Xml(error)
    }
}

impl From<std::io::Error> for AppError {
    fn from(error: std::io::Error) -> Self {
        Self::IOError(error)
    }
}

impl From<csv::Error> for AppError {
    fn from(error: csv::Error) -> Self {
        Self::CSVError(error)
    }
}


#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// XML file path
    #[arg(short, long)]
    input: PathBuf,

    /// Output file path
    #[arg(short, long)]
    output: PathBuf,
}

fn main() -> Result<(), AppError> {
    let args = Args::parse();


    if !args.input.is_file() {
        return Err(AppError::InvalidInputPath)
    }

    let is_input_file_xml = args.input
        .extension()
        .and_then(OsStr::to_str)
        .map_or(false, |ext| ext == "xml");
    if !is_input_file_xml {
        return Err(AppError::InvalidInputPath);
    }

    let mut reader: Reader<std::io::BufReader<std::fs::File>> = Reader::from_file(args.input)?;
    reader.trim_text(true);

    let mut buf: Vec<u8> = Vec::new();
    let mut data: Vec<BiologicalData> = Vec::new();
    let max_data_to_flush = 10000_usize;

    let mut acumulated_data_extracted = 0;
    let output_file = std::fs::OpenOptions::new()
        .write(true)
        .append(true)
        .create(true)
        .open(args.output)?;
    let mut file_writer = csv::WriterBuilder::new().from_writer(output_file);

    let start_extracting_time = Instant::now();
    loop {
        let event = reader.read_event_into(&mut buf)?;

        match event {
            Event::Start(element) => {
                if element.name().as_ref() != b"entry" {
                    continue;
                }
                let biological_data = extract_entry(element, &mut reader);
                data.push(biological_data);
            },

            Event::Eof => {
                buf.clear();
                break
            },
            _ => (),
        }

        if data.len() >= max_data_to_flush {
            acumulated_data_extracted += data.len();
            println!("Acumulated data extracted: {} data points", acumulated_data_extracted);
            save_data(&mut data, &mut file_writer)?;
        }
        buf.clear();
    }
    
    acumulated_data_extracted += data.len();
    save_data(&mut data, &mut file_writer)?;

    let ended_extracting_time = start_extracting_time.elapsed();
    
    println!("It took {:?} to extract {} data points", ended_extracting_time, acumulated_data_extracted);
    
    Ok(())
}

fn save_data(data: &mut Vec<BiologicalData>, writter: &mut csv::Writer<File>) -> Result<(), AppError> {
    for d in data.iter() {
        let enzyme = if d.is_enzyme() {
            "1"
        } else {
            "0"
        };
        writter.write_record([d.accession(), d.sequence(), enzyme])?;
    }
    data.clear();
    Ok(())
}