from scripts.read_rd03d_uart import bytes_to_hex, bytes_to_printable


def test_bytes_to_hex_formats_uppercase_octets():
    assert bytes_to_hex(bytes([0x00, 0x0A, 0xFF])) == "00 0A FF"


def test_bytes_to_printable_replaces_control_bytes():
    assert bytes_to_printable(b"A\x00\nZ") == "A..Z"
