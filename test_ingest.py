import unittest
import os
from ingest import extract_pdf_pages, resolve_pdf_path, resolve_source_refs

class TestDocumentIngestion(unittest.TestCase):
    def test_resolve_pdf_path(self):
        # 1. Clean path
        p1 = resolve_pdf_path("papers/SoliMix_Rosenstock2021.pdf")
        self.assertTrue(os.path.exists(p1), f"Path not found: {p1}")

        # 2. Path with page annotation in parentheses
        p2 = resolve_pdf_path("SOLIQUA_LBL.pdf (Prescribing Information, page 4)")
        self.assertTrue(os.path.exists(p2), f"Path not found: {p2}")

        # 3. Path with extra comment
        p3 = resolve_pdf_path("papers/Tabak2020_PSM.pdf (plus LixiLan-L abstract and SoliMix)")
        self.assertTrue(os.path.exists(p3), f"Path not found: {p3}")

    def test_page_annotation_is_not_a_source(self):
        for citation in (
            "SOLIQUA_LBL.pdf (Prescribing Information, page 4)",
            "SOLIQUA_LBL.pdf (see page 4)",
        ):
            refs = resolve_source_refs(citation)
            self.assertEqual(len(refs), 1)
            self.assertEqual(refs[0].label, "SOLIQUA_LBL.pdf")

    def test_extract_solimix_pages_and_vector_text(self):
        path = resolve_pdf_path("papers/SoliMix_Rosenstock2021.pdf")
        pages = extract_pdf_pages(path)
        
        self.assertGreaterEqual(len(pages), 7)
        
        # SoliMix Figure 1B on Page 5
        p5_text = pages[5]["text"]
        self.assertIn("-0.2 (-0.4, -0.1)", p5_text)
        
        # SoliMix Figure 2 on Page 6
        p6_text = pages[6]["text"]
        self.assertIn("42.2", p6_text)
        self.assertIn("1.65 (1.25, 2.19)", p6_text)
        
        # SoliMix Figure 3 on Page 7
        p7_text = pages[7]["text"]
        self.assertIn("0.45 (0.28, 0.73)", p7_text)

    def test_extract_lixilano_text_and_tables(self):
        path = resolve_pdf_path("papers/LixiLan-O_Rosenstock2016.pdf")
        pages = extract_pdf_pages(path)
        
        # Page 13 contains -1.63% and -1.34%
        p13_text = pages[13]["text"]
        self.assertIn("1.63%", p13_text)
        self.assertIn("1.34%", p13_text)
        
        # Page 29 contains Table 2
        p29_tables = pages[29]["tables"]
        self.assertTrue(len(p29_tables) > 0 or "Table 2" in pages[29]["text"])

    def test_scanned_lbl_detection(self):
        path = resolve_pdf_path("SOLIQUA_LBL.pdf")
        pages = extract_pdf_pages(path)
        
        # SOLIQUA_LBL is an image-based scanned PDF
        self.assertEqual(len(pages), 4)
        for p in range(1, 5):
            self.assertFalse(pages[p]["has_native_text"], f"Page {p} should be detected as scanned image")
            self.assertTrue(pages[p]["needs_vision"], f"Page {p} should be sent to vision")
            self.assertIsNotNone(pages[p]["image_bytes"], f"Page {p} should provide rendered image bytes")

    def test_pdf_control_characters_are_sanitized(self):
        path = resolve_pdf_path("papers/SoliMix_Rosenstock2021.pdf")
        pages = extract_pdf_pages(path)
        page_one = pages[1]["text"]
        self.assertIn("mean difference -0.2%", page_one)
        self.assertNotIn("\x01", page_one)

    def test_raster_page_with_thin_header_needs_vision(self):
        path = resolve_pdf_path("papers/LixiLan-L_Aroda2016_abstract.pdf")
        pages = extract_pdf_pages(path)
        # Page 2 carries the abstract's raster body with only a thin native
        # header, so reading native text alone would miss the table.
        self.assertTrue(pages[2]["needs_vision"])
        self.assertIsNotNone(pages[2]["image_bytes"])

if __name__ == "__main__":
    unittest.main()
