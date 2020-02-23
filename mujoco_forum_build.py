#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Builds the Jitterbug XML file for upload to the MuJoCo forums

The MuJoCo forums only support uplodaing a single XML file, however the
Jitterbug model definition includes several sub-files. This script copies the
contents of any included file into the master file so we can upload a single
XML to the MuJoCo website.
"""

assert __name__ == "__main__",\
    "Please run this file rather than importing it"

import os
import xml.etree.cElementTree as ET

base_path = "jitterbug_dmc"
main_file = os.path.join(base_path, "jitterbug.xml")
out_file = os.path.join(".", "jitterbug_mujoco.xml")

if os.path.exists(main_file):

    tree = ET.parse(main_file)
    root = tree.getroot()

    # Build parent map
    parents = {c:p for p in tree.iter() for c in p}

    for include in root.findall("include"):
        # Copy referenced file's contents to the current document
        parents[include].append(
            ET.parse(
                os.path.join(base_path, include.get("file"))
            ).getroot().getchildren()[0]
        )
        root.remove(include)

    tree.write(out_file)
    print("Saved to {}".format(out_file))


print("Done")
