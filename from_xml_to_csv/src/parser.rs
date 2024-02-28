use quick_xml::{events::Event, Reader};

use crate::data::BiologicalData;

pub fn extract_entry(_entry: quick_xml::events::BytesStart<'_>, reader: &mut Reader<std::io::BufReader<std::fs::File>>) -> BiologicalData {
    let mut buf: Vec<u8> = Vec::new();
    let mut accession: Vec<u8> = Vec::new();
    let mut sequence: Vec<u8> = Vec::new();
    let mut is_enzyme = false;
    loop {
        let event = reader.read_event_into(&mut buf).unwrap();
        match event {
            Event::Start(ref element) => {
                let name = element.name();
                if name.as_ref() == b"accession" && accession.is_empty() {
                    // We are sure the next event is exactly the text inside <accesion></accesion>
                    reader
                        .read_event_into(&mut accession)
                        .expect("Should have gotten text value from element 'accession'");
                }
                if name.as_ref() == b"sequence" {
                    // We are sure the next event is exactly the text inside <sequence></sequence>
                    reader
                        .read_event_into(&mut sequence)
                        .expect("Should have gotten text value from element 'accession'");
                }
                if name.as_ref() == b"comment" {
                    for attr in element.attributes().map(|e| e.unwrap()) {
                        if attr.value.as_ref() == b"catalytic activity" {
                            is_enzyme = true;
                        }
                        
                    }
                }
            }
            Event::End(e) => {
                let name = e.name();
                if name.as_ref() == b"entry" {
                    break;
                }
            },
            _ => {}
        }
        buf.clear();
    }

    let accession = String::from_utf8(accession).unwrap();
    let sequence = String::from_utf8(sequence).unwrap();
    BiologicalData::new(accession, sequence, is_enzyme)
}
