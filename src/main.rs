extern crate Fast_QR;
use Fast_QR::QR;
use Fast_QR::QR_e;

fn print_qr(qr: &QR) {
	let border: i32 = 4;
	for y in -border .. qr.size() + border {
		for x in -border .. qr.size() + border {
			let c: char = if qr.get_module(x, y) { '█' } else { ' ' };
			print!("{0}{0}", c);
		}
		println!();
	}
	println!();
}
fn do_basic_demo() {
	let text: &'static str = "Kerim Büyükakyüz";
	let errcorlvl: QR_e = QR_e::Low;
	let qr: QR = QR::encode_s(text, errcorlvl).unwrap();
	print_qr(&qr);
}
pub fn main() {
	do_basic_demo();
}