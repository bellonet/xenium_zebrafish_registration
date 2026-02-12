"""
Inspect OME-TIFF file structure to understand pyramid levels and series.
"""
import sys
import tifffile
from pathlib import Path


def inspect_tiff(filepath):
    """Detailed inspection of TIFF file structure."""
    print("=" * 80)
    print(f"Inspecting: {filepath}")
    print("=" * 80)

    with tifffile.TiffFile(filepath) as tif:
        print(f"\n{'FILE INFO':=^80}")
        print(f"Is OME: {tif.is_ome}")
        print(f"Is ImageJ: {tif.is_imagej}")
        print(f"Is Shaped: {tif.is_shaped}")
        print(f"Byte order: {tif.byteorder}")

        print(f"\n{'PAGES INFO':=^80}")
        print(f"Number of pages: {len(tif.pages)}")

        # Show first few pages
        print(f"\nFirst 10 pages:")
        for i, page in enumerate(tif.pages[:10]):
            print(f"  Page {i}:")
            print(f"    Shape: {page.shape}")
            print(f"    Dtype: {page.dtype}")
            print(f"    Compression: {page.compression}")
            if hasattr(page, 'tags') and 'ImageDescription' in page.tags:
                desc = page.tags['ImageDescription'].value[:100]
                print(f"    Description: {desc}...")

        if len(tif.pages) > 10:
            print(f"  ... and {len(tif.pages) - 10} more pages")

        print(f"\n{'SERIES INFO':=^80}")
        print(f"Number of series: {len(tif.series)}")

        for i, series in enumerate(tif.series):
            print(f"\nSeries {i}:")
            print(f"  Name: {series.name}")
            print(f"  Shape: {series.shape}")
            print(f"  Dtype: {series.dtype}")
            print(f"  Axes: {series.axes}")
            print(f"  Kind: {series.kind}")

            # Check for pyramid levels
            if hasattr(series, 'levels'):
                print(f"  Number of pyramid levels: {len(series.levels)}")
                for level_idx, level in enumerate(series.levels):
                    print(f"    Level {level_idx}:")
                    print(f"      Shape: {level.shape}")
                    print(f"      Axes: {level.axes}")
                    if hasattr(level, 'pages'):
                        print(f"      Pages: {len(level.pages)}")
            else:
                print(f"  No pyramid levels attribute")

            # Try to get keyframe
            if hasattr(series, 'keyframe'):
                print(f"  Keyframe shape: {series.keyframe.shape}")

        print(f"\n{'TRYING TO READ DATA':=^80}")

        # Try different methods to read the data
        methods = [
            ("Full series 0", lambda: tif.asarray(series=0)),
            ("Series 0, key=0", lambda: tif.asarray(series=0, key=0)),
            ("Just key=0", lambda: tif.asarray(key=0)),
            ("Pages[0]", lambda: tif.pages[0].asarray()),
        ]

        for method_name, method_func in methods:
            try:
                print(f"\nTrying: {method_name}")
                data = method_func()
                print(f"  ✓ Success! Shape: {data.shape}, Dtype: {data.dtype}")
            except Exception as e:
                print(f"  ✗ Failed: {type(e).__name__}: {e}")

        # Try to read pyramid levels if they exist
        if len(tif.series) > 0 and hasattr(tif.series[0], 'levels'):
            print(f"\n{'TESTING PYRAMID LEVELS':=^80}")
            for level_idx in range(len(tif.series[0].levels)):
                try:
                    print(f"\nTrying level {level_idx}:")
                    data = tif.asarray(series=0, level=level_idx)
                    print(f"  ✓ Success! Shape: {data.shape}, Dtype: {data.dtype}")
                except Exception as e:
                    print(f"  ✗ Failed: {type(e).__name__}: {e}")

        # Alternative: try reading pages as channels
        print(f"\n{'TESTING PAGE-BASED READING':=^80}")
        print(f"Attempting to read first 4 pages as channels:")
        for page_idx in range(min(4, len(tif.pages))):
            try:
                print(f"\nPage {page_idx}:")
                data = tif.pages[page_idx].asarray()
                print(f"  ✓ Shape: {data.shape}, Dtype: {data.dtype}")

                # Check for subifds (pyramid)
                page = tif.pages[page_idx]
                if hasattr(page, 'subifds') and page.subifds:
                    print(f"  SubIFDs found: {len(page.subifds)}")
                    for subid_idx, subifd in enumerate(page.subifds):
                        try:
                            subdata = subifd.asarray()
                            print(f"    SubIFD {subid_idx}: Shape {subdata.shape}")
                        except Exception as e:
                            print(f"    SubIFD {subid_idx}: Failed - {e}")

            except Exception as e:
                print(f"  ✗ Failed: {type(e).__name__}: {e}")

        print(f"\n{'OME METADATA':=^80}")
        if tif.is_ome and tif.ome_metadata:
            ome_xml = tif.ome_metadata
            print(f"OME-XML length: {len(ome_xml)} characters")
            print(f"\nFirst 500 characters:")
            print(ome_xml[:500])
            print("...")

            # Try to parse with xml
            try:
                import xml.etree.ElementTree as ET
                root = ET.fromstring(ome_xml)
                print(f"\nRoot tag: {root.tag}")

                # Find all Image elements
                ns = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
                images = root.findall('.//ome:Image', ns)
                print(f"Number of Image elements: {len(images)}")

                for img_idx, img in enumerate(images[:3]):
                    print(f"\nImage {img_idx}:")
                    name = img.get('Name')
                    print(f"  Name: {name}")

                    pixels = img.find('.//ome:Pixels', ns)
                    if pixels is not None:
                        print(f"  SizeX: {pixels.get('SizeX')}")
                        print(f"  SizeY: {pixels.get('SizeY')}")
                        print(f"  SizeZ: {pixels.get('SizeZ')}")
                        print(f"  SizeC: {pixels.get('SizeC')}")
                        print(f"  SizeT: {pixels.get('SizeT')}")

                        channels = pixels.findall('.//ome:Channel', ns)
                        print(f"  Channels: {len(channels)}")
                        for ch_idx, ch in enumerate(channels):
                            print(f"    Channel {ch_idx}: {ch.get('Name', 'N/A')}")

            except Exception as e:
                print(f"Could not parse OME-XML: {e}")

        print("\n" + "=" * 80)
        print("SUMMARY & RECOMMENDATIONS")
        print("=" * 80)

        # Provide recommendations
        if hasattr(tif.series[0], 'levels') and len(tif.series[0].levels) > 1:
            print("\n✓ This file HAS pyramid levels via series.levels")
            print(f"  Available levels: {len(tif.series[0].levels)}")
            print("  Recommended reading method:")
            print("    data = tifffile.imread(path, series=0, level=N)")
        elif len(tif.pages) > 0 and hasattr(tif.pages[0], 'subifds') and tif.pages[0].subifds:
            print("\n✓ This file HAS pyramid levels via SubIFDs")
            print(f"  SubIFDs per page: {len(tif.pages[0].subifds)}")
            print("  Recommended reading method:")
            print("    page = tif.pages[channel_idx]")
            print("    full_res = page.asarray()")
            print("    pyramid_level_1 = page.subifds[0].asarray()")
        else:
            print("\n⚠ No pyramid levels detected!")
            print("  This file may not have pyramids, or they're stored differently")
            print("  Recommended reading method:")
            print("    data = tifffile.imread(path, key=channel_idx)")


if __name__ == "__main__":
    # Default path if no argument provided
    default_path = "../output-XETG00046__0043921__Region_1__20250620__084504/morphology_focus/morphology_focus_0000.ome.tif"

    if len(sys.argv) < 2:
        # Try the default path
        if Path(default_path).exists():
            print(f"No path provided, using default: {default_path}\n")
            filepath = default_path
        else:
            # Look in current directory
            tif_files = list(Path('.').glob("*.tif")) + list(Path('.').glob("*.tiff"))
            if tif_files:
                filepath = str(tif_files[0])
                print(f"No path provided, found: {filepath}\n")
            else:
                print("Usage: python inspect_tiff.py [path_to_tiff_file_or_directory]")
                print("\nNo path provided and no .tif files found in current directory")
                print("\nExample:")
                print("  python inspect_tiff.py morphology_focus_0000.ome.tif")
                print("  python inspect_tiff.py ../output-folder/morphology_focus/")
                sys.exit(1)
    else:
        filepath = sys.argv[1]

    path_obj = Path(filepath)

    # If it's a directory, find .tif files in it
    if path_obj.is_dir():
        tif_files = list(path_obj.glob("*.tif")) + list(path_obj.glob("*.tiff"))
        if not tif_files:
            print(f"Error: No .tif or .tiff files found in directory: {filepath}")
            sys.exit(1)

        print(f"Found {len(tif_files)} TIFF file(s) in {filepath}:")
        for i, f in enumerate(tif_files):
            print(f"  {i + 1}. {f.name}")

        # Inspect the first one
        filepath = str(tif_files[0])
        print(f"\nInspecting first file: {filepath}")
        print()
    elif not path_obj.exists():
        print(f"Error: File or directory not found: {filepath}")
        sys.exit(1)
    else:
        filepath = str(path_obj)

    inspect_tiff(filepath)