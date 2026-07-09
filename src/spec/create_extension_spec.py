# -*- coding: utf-8 -*-
import os.path

from pynwb.spec import NWBNamespaceBuilder, export_spec, NWBGroupSpec, NWBAttributeSpec, NWBDatasetSpec


def main():
    # these arguments were auto-generated from your cookiecutter inputs
    ns_builder = NWBNamespaceBuilder(
        doc="""Extension for defining neural probes in the probeinterface format""",
        name="""ndx-probeinterface""",
        version="""0.3.0""",
        author=["Alessio Buccino", "Kyu Hyun Lee", "Geeling Chau", "Heberto Mayorquin"],
        contact=["alessiop.buccino@gmail.com", "kyuhyun9056@gmail.com", "gchau@caltech.edu", "h.mayorquin@gmail.com"],
    )

    # ProbeModel extends the core DeviceModel, Probe extends the core Device.
    # Both are shipped by pynwb 4.0, so the probe geometry (catalogue) and the
    # physical instance are separated onto the same rails core already provides.
    ns_builder.include_type(data_type="DeviceModel", namespace="core")
    ns_builder.include_type(data_type="Device", namespace="core")
    ns_builder.include_namespace("hdmf-common")

    # Per-contact geometry only. This table is owned by ProbeModel (the catalogue
    # object), so identical probes reference one shared table. There is no
    # device_channel_index column: the channel-to-contact link is by contact_id,
    # not by a device index, so wiring is not part of probe serialization.
    contacts_table = NWBGroupSpec(
        doc="Neural probe contacts according to probeinterface specification",
        datasets=[
            NWBDatasetSpec(
                name="contact_position",
                doc="position of the contact",
                dtype="float",
                dims=[["num_contacts", "x, y"], ["num_contacts", "x, y, z"]],
                shape=[[None, 2], [None, 3]],
                neurodata_type_inc="VectorData",
            ),
            NWBDatasetSpec(
                name="contact_shape",
                doc="shape of the contact; e.g. 'circle'",
                dtype="text",
                neurodata_type_inc="VectorData",
            ),
            NWBDatasetSpec(
                name="contact_id",
                doc="unique ID of the contact; the stable link key",
                dtype="text",
                neurodata_type_inc="VectorData",
            ),
            NWBDatasetSpec(
                name="shank_id",
                doc="shank ID of the contact",
                dtype="text",
                neurodata_type_inc="VectorData",
                quantity="?",
            ),
            NWBDatasetSpec(
                name="contact_plane_axes",
                doc="the axes defining the contact plane",
                dtype="float",
                dims=[["num_contacts", "v1, v2", "x,y"], ["num_contacts", "v1,v2", "x, y, z"]],
                shape=[[None, 2, 2], [None, 2, 3]],
                neurodata_type_inc="VectorData",
                quantity="?",
            ),
            NWBDatasetSpec(
                name="radius",
                doc="radius of a circular contact",
                dtype="float",
                neurodata_type_inc="VectorData",
                quantity="?",
            ),
            NWBDatasetSpec(
                name="width",
                doc="width of a rectangular or square contact",
                dtype="float",
                neurodata_type_inc="VectorData",
                quantity="?",
            ),
            NWBDatasetSpec(
                name="height",
                doc="height of a rectangular contact",
                dtype="float",
                neurodata_type_inc="VectorData",
                quantity="?",
            ),
        ],
        neurodata_type_inc="DynamicTable",
        neurodata_type_def="ContactsTable",
    )

    # ProbeModel is the catalogue object: reusable, reconstructable geometry plus
    # identity (manufacturer, model_number, description all inherited from
    # DeviceModel). One ProbeModel is stored per distinct model, and identical
    # physical probes share it.
    probe_model = NWBGroupSpec(
        doc="Neural probe model (catalogue geometry + identity), extends core DeviceModel",
        attributes=[
            NWBAttributeSpec(
                name="ndim",
                doc="dimension of the probe (2 or 3)",
                dtype="int",
                required=True,
                default_value=2,
            ),
            NWBAttributeSpec(
                name="unit",
                doc="SI unit used to define the probe geometry; e.g. 'micrometer'.",
                dtype="text",
                required=True,
                default_value="micrometer",
            ),
        ],
        neurodata_type_inc="DeviceModel",
        neurodata_type_def="ProbeModel",
        groups=[
            NWBGroupSpec(
                doc="Per-contact geometry for this probe model",
                neurodata_type_inc="ContactsTable",
                quantity=1,
            )
        ],
        datasets=[
            NWBDatasetSpec(
                name="planar_contour",
                doc="The planar polygon that outlines the probe contour.",
                dtype="float",
                dims=[["num_points", "x"], ["num_points", "x, y"], ["num_points", "x, y, z"]],
                shape=[[None, 1], [None, 2], [None, 3]],
                quantity="?",
            )
        ],
    )

    # Probe is the physical instance in an experiment. serial_number and the
    # model link are both provided by core Device; only the probeinterface
    # instance-level annotations are added here.
    probe = NWBGroupSpec(
        doc="Physical neural probe instance, extends core Device with a link to a ProbeModel",
        attributes=[
            NWBAttributeSpec(
                name="annotations",
                doc="instance-level probeinterface annotations, JSON-encoded",
                dtype="text",
                required=False,
            ),
        ],
        neurodata_type_inc="Device",
        neurodata_type_def="Probe",
    )

    new_data_types = [probe_model, probe, contacts_table]

    # export the spec to yaml files in the spec folder
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "spec"))
    export_spec(ns_builder, new_data_types, output_dir)
    print("Spec files generated. Please make sure to rerun `pip install .` to load the changes.")


if __name__ == "__main__":
    # usage: python create_extension_spec.py
    main()
