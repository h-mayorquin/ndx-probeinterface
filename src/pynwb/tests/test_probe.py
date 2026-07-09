import datetime
import json

import numpy as np
import probeinterface as pi

from pynwb import NWBHDF5IO, NWBFile
from pynwb.device import Device, DeviceModel
from pynwb.testing import TestCase, remove_test_file

from ndx_probeinterface import Probe, ProbeModel, ContactsTable

# annotation keys promoted to first-class fields, so they are not expected back
# inside the round-tripped annotations dict.
PROMOTED_KEYS = ["name", "manufacturer", "model_name", "serial_number"]


def set_up_nwbfile():
    return NWBFile(
        session_description="session_description",
        identifier="identifier",
        session_start_time=datetime.datetime.now(datetime.timezone.utc),
    )


def create_single_shank_probe():
    probe = pi.generate_linear_probe()
    probe.annotate(name="Single-shank", custom_key="custom annotation")
    probe.set_contact_ids([f"c{i}" for i in range(probe.get_contact_count())])
    return probe


def create_multi_shank_probe():
    probe = pi.generate_multi_shank()
    probe.annotate(name="Multi-shank", custom_key="custom annotation")
    probe.set_contact_ids([f"cm{i}" for i in range(probe.get_contact_count())])
    return probe


def filtered_annotations(annotations):
    return {key: value for key, value in annotations.items() if key not in PROMOTED_KEYS}


def add_probe_to_nwbfile(nwbfile, ndx_probe):
    """Add a Probe and its (possibly shared) ProbeModel to an NWB file."""
    if ndx_probe.model.name not in nwbfile.device_models:
        nwbfile.add_device_model(ndx_probe.model)
    nwbfile.add_device(ndx_probe)


class TestFromProbeInterface(TestCase):
    """The from_probeinterface split into ProbeModel (catalogue) + Probe (instance)."""

    def setUp(self):
        self.probe0 = create_single_shank_probe()
        self.probe1 = create_multi_shank_probe()

    def test_probe_is_device_with_probemodel(self):
        ndx_probe = Probe.from_probeinterface(self.probe0)[0]
        self.assertIsInstance(ndx_probe, Probe)
        self.assertIsInstance(ndx_probe, Device)
        self.assertIsInstance(ndx_probe.model, ProbeModel)
        self.assertIsInstance(ndx_probe.model, DeviceModel)

    def test_geometry_lives_on_the_model(self):
        ndx_probe = Probe.from_probeinterface(self.probe0)[0]
        contacts_table = ndx_probe.model.contacts_table
        self.assertIsInstance(contacts_table, ContactsTable)
        probe_array = self.probe0.to_numpy()
        np.testing.assert_array_equal(
            contacts_table["contact_position"][:], self.probe0.contact_positions
        )
        np.testing.assert_array_equal(
            contacts_table["contact_shape"][:], probe_array["contact_shapes"]
        )

    def test_no_device_channel_index_column(self):
        # device_channel wiring is not part of probe serialization
        self.probe0.set_device_channel_indices(np.arange(self.probe0.get_contact_count()))
        ndx_probe = Probe.from_probeinterface(self.probe0)[0]
        colnames = ndx_probe.model.contacts_table.colnames
        self.assertFalse(any("device_channel" in name for name in colnames))

    def test_multi_shank_keeps_shank_id(self):
        ndx_probe = Probe.from_probeinterface(self.probe1)[0]
        contacts_table = ndx_probe.model.contacts_table
        np.testing.assert_array_equal(contacts_table["shank_id"][:], self.probe1.shank_ids)

    def test_instance_annotations_exclude_promoted_keys(self):
        ndx_probe = Probe.from_probeinterface(self.probe0)[0]
        self.assertDictEqual(
            json.loads(ndx_probe.annotations), filtered_annotations(self.probe0.annotations)
        )


class TestProbeModelDeduplication(TestCase):
    """A ProbeGroup of identical probes shares one ProbeModel; distinct probes do not."""

    def test_identical_probes_share_one_model(self):
        probegroup = pi.ProbeGroup()
        probegroup.add_probe(pi.generate_linear_probe())
        probegroup.add_probe(pi.generate_linear_probe())

        ndx_probes = Probe.from_probeinterface(probegroup)
        self.assertEqual(len(ndx_probes), 2)
        self.assertIs(ndx_probes[0].model, ndx_probes[1].model)
        self.assertNotEqual(ndx_probes[0].name, ndx_probes[1].name)

    def test_distinct_probes_get_distinct_models(self):
        probegroup = pi.ProbeGroup()
        probegroup.add_probe(create_single_shank_probe())
        probegroup.add_probe(create_multi_shank_probe())

        ndx_probes = Probe.from_probeinterface(probegroup)
        self.assertEqual(len(ndx_probes), 2)
        self.assertIsNot(ndx_probes[0].model, ndx_probes[1].model)


class TestToProbeInterface(TestCase):
    """Round-trip back to probeinterface is id-based and drops device wiring."""

    def setUp(self):
        self.probe0 = create_single_shank_probe()
        self.probe1 = create_multi_shank_probe()

    def test_geometry_and_annotations_roundtrip(self):
        for probe in (self.probe0, self.probe1):
            ndx_probe = Probe.from_probeinterface(probe)[0]
            reconstructed = ndx_probe.to_probeinterface()
            np.testing.assert_array_equal(probe.to_numpy(), reconstructed.to_numpy())
            self.assertDictEqual(probe.annotations, reconstructed.annotations)

    def test_contact_ids_roundtrip(self):
        ndx_probe = Probe.from_probeinterface(self.probe0)[0]
        reconstructed = ndx_probe.to_probeinterface()
        np.testing.assert_array_equal(reconstructed.contact_ids, self.probe0.contact_ids)

    def test_device_channel_indices_not_reconstructed(self):
        self.probe0.set_device_channel_indices(np.arange(self.probe0.get_contact_count()))
        ndx_probe = Probe.from_probeinterface(self.probe0)[0]
        reconstructed = ndx_probe.to_probeinterface()
        self.assertIsNone(reconstructed.device_channel_indices)


class TestProbeRoundtripNWB(TestCase):
    """Full write/read of the ProbeModel + Probe through an NWB HDF5 file."""

    def setUp(self):
        self.probe0 = create_single_shank_probe()
        self.probe1 = create_multi_shank_probe()
        self.path = "test_probeinterface_roundtrip.nwb"

    def tearDown(self):
        remove_test_file(self.path)

    def _roundtrip(self, pi_probe_or_group):
        nwbfile = set_up_nwbfile()
        ndx_probes = Probe.from_probeinterface(pi_probe_or_group)
        for ndx_probe in ndx_probes:
            add_probe_to_nwbfile(nwbfile, ndx_probe)

        with NWBHDF5IO(self.path, mode="w") as io:
            io.write(nwbfile)

        with NWBHDF5IO(self.path, mode="r", load_namespaces=True) as io:
            read_nwbfile = io.read()
            for ndx_probe in ndx_probes:
                read_probe = read_nwbfile.devices[ndx_probe.name]
                self.assertIsInstance(read_probe, Probe)
                self.assertIn(ndx_probe.model.name, read_nwbfile.device_models)
                self.assertContainerEqual(ndx_probe, read_probe)

    def test_roundtrip_single_shank(self):
        self._roundtrip(self.probe0)

    def test_roundtrip_multi_shank(self):
        self._roundtrip(self.probe1)

    def test_roundtrip_probegroup_with_dedup(self):
        probegroup = pi.ProbeGroup()
        probegroup.add_probe(pi.generate_linear_probe())
        probegroup.add_probe(pi.generate_linear_probe())

        nwbfile = set_up_nwbfile()
        ndx_probes = Probe.from_probeinterface(probegroup)
        for ndx_probe in ndx_probes:
            add_probe_to_nwbfile(nwbfile, ndx_probe)

        # dedup: two probes, one model registered
        self.assertEqual(len(nwbfile.device_models), 1)

        with NWBHDF5IO(self.path, mode="w") as io:
            io.write(nwbfile)
        with NWBHDF5IO(self.path, mode="r", load_namespaces=True) as io:
            read_nwbfile = io.read()
            self.assertEqual(len(read_nwbfile.device_models), 1)
            self.assertEqual(len(read_nwbfile.devices), 2)
