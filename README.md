# RQR
Rapid QR Code Generator

Rapid QR Code Generator is the first generator of its kind. It compiles to native machine-executable code with **zero dependencies**. The generator is guaranteed to generate a valid QR Code given its string input is less than 4296 characters (theoretical limit) and the string is UTF-8 encoded. This generation is guaranteed by enforced memory safety without relying on garbage collection, namely, all its references are pointing to a valid memory in RAM by means of static lifetimes. Therefore, it is up to 20% faster than any contemporary QR Code Generator. Implemented Reed–Solomon error correction makes up to 30% error correction in high, available with the options medium and low.


Example: 

>let text: &'static str = "Kerim Büyükakyüz - this is a string literal!";
>process()

You can scan the QR Code to check the validity of the output 

RQR: 

![ex](https://user-images.githubusercontent.com/99087793/177833176-ecd3d71c-a0b4-4ce6-8d7f-3dcd622c3a13.png)
