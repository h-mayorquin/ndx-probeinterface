from typing import Union, List
import numpy as np
import json
from probeinterface import Probe, ProbeGroup

unit_map = {
    "um": "micrometer",
    "mm": "millimeter",
    "m": "meter",
}
inverted_unit_map = {v: k for k, v in unit_map.items()}

# annotations promoted to first-class fields on ProbeModel / Probe, so they are
# not re-stored in the JSON annotations blob.
_PROMOTED_ANNOTATIONS = ("name", "serial_number", "model_name", "manufacturer")

# core DeviceModel requires a manufacturer; probeinterface probes generated in
# code carry none, so a sentinel is stored and mapped back to None on the way out.
_UNKNOWN_MANUFACTURER = "unknown"


def from_probeinterface(probe_or_probegroup: Union[Probe, ProbeGroup]) -> List["Probe"]:
    """
    Construct ndx-probeinterface Probe instances from a probeinterface.Probe or ProbeGroup.

    Each returned Probe (an extension of the core Device) carries a link to a
    ProbeModel (an extension of the core DeviceModel) that holds the geometry.
    Physically identical probes share a single ProbeModel instance, so a
    ProbeGroup of N identical probes yields one ProbeModel and N Probe devices.

    Parameters
    ----------
    probe_or_probegroup: Probe or ProbeGroup
        Probe or ProbeGroup to convert.

    Returns
    -------
    probes: list
        The list of ndx-probeinterface Probe devices, each with ``.model`` set.
    """
    assert isinstance(
        probe_or_probegroup, (Probe, ProbeGroup)
    ), f"The input must be a Probe or ProbeGroup, not {type(probe_or_probegroup)}"
    if isinstance(probe_or_probegroup, Probe):
        pi_probes = [probe_or_probegroup]
    else:
        pi_probes = probe_or_probegroup.probes

    models_by_signature = {}
    ndx_probes = []
    for index, pi_probe in enumerate(pi_probes):
        signature = _probe_model_signature(pi_probe)
        probe_model = models_by_signature.get(signature)
        if probe_model is None:
            probe_model = _probe_to_probe_model(pi_probe, model_index=len(models_by_signature))
            models_by_signature[signature] = probe_model
        ndx_probes.append(_probe_to_device(pi_probe, probe_model, instance_index=index))
    return ndx_probes


def to_probeinterface(ndx_probe) -> Probe:
    """
    Construct a probeinterface.Probe from an ndx-probeinterface Probe.

    Geometry, identity, and contact_ids round-trip losslessly. The
    channel-to-contact wiring (``device_channel_indices``) is intentionally not
    reconstructed: it is recovered from ``contact_id`` where actually needed.

    Parameters
    ----------
    ndx_probe: ndx_probeinterface.Probe

    Returns
    -------
    Probe: probeinterface.Probe
    """
    probe_model = ndx_probe.model
    ndim = probe_model.ndim
    unit = inverted_unit_map[probe_model.unit]
    contacts_table = _get_contacts_table(probe_model)

    positions = contacts_table["contact_position"][:]
    shapes = contacts_table["contact_shape"][:]
    contact_ids = contacts_table["contact_id"][:]

    plane_axes = None
    if "contact_plane_axes" in contacts_table.colnames:
        plane_axes = contacts_table["contact_plane_axes"][:]
    shank_ids = None
    if "shank_id" in contacts_table.colnames:
        shank_ids = contacts_table["shank_id"][:]

    shape_params = None
    possible_shape_keys = ["radius", "width", "height"]
    for shape_key in possible_shape_keys:
        if shape_key in contacts_table.colnames:
            if shape_params is None:
                shape_params = [{} for _ in range(len(contacts_table))]
            for i in range(len(contacts_table)):
                shape_params[i][shape_key] = contacts_table[shape_key][i]

    manufacturer = probe_model.manufacturer
    if manufacturer == _UNKNOWN_MANUFACTURER:
        manufacturer = None
    probeinterface_probe = Probe(
        ndim=ndim,
        si_units=unit,
        name=ndx_probe.name,
        serial_number=ndx_probe.serial_number,
        manufacturer=manufacturer,
        model_name=probe_model.model_number,
    )
    probeinterface_probe.set_contacts(
        positions=positions,
        shapes=shapes,
        shape_params=shape_params,
        plane_axes=plane_axes,
        shank_ids=shank_ids,
    )
    probeinterface_probe.set_contact_ids(contact_ids=contact_ids)
    if probe_model.planar_contour is not None:
        probeinterface_probe.set_planar_contour(probe_model.planar_contour[:])
    if ndx_probe.annotations is not None:
        probeinterface_probe.annotate(**json.loads(ndx_probe.annotations))

    return probeinterface_probe


def _probe_model_signature(probe: Probe):
    """A hashable signature that is equal for physically identical probes."""
    contacts_arr = probe.to_numpy()
    return (
        probe.manufacturer,
        probe.model_name,
        probe.ndim,
        probe.si_units,
        np.asarray(probe.contact_positions).tobytes(),
        tuple(contacts_arr["contact_shapes"].tolist()),
        None if probe.probe_planar_contour is None else np.asarray(probe.probe_planar_contour).tobytes(),
    )


def _contact_ids_or_default(probe: Probe) -> List[str]:
    """probeinterface leaves contact_ids as empty strings when unset; synthesize stable ids."""
    contact_ids = probe.contact_ids
    if contact_ids is None or all(str(contact_id) == "" for contact_id in contact_ids):
        return [str(i) for i in range(probe.get_contact_count())]
    return [str(contact_id) for contact_id in contact_ids]


def _probe_to_probe_model(probe: Probe, model_index: int):
    from pynwb import get_class

    ProbeModel = get_class("ProbeModel", "ndx-probeinterface")
    ContactsTable = get_class("ContactsTable", "ndx-probeinterface")

    contact_positions = probe.contact_positions
    contact_plane_axes = probe.contact_plane_axes
    contact_ids = _contact_ids_or_default(probe)
    contacts_arr = probe.to_numpy()

    shape_keys = []
    for shape_params in probe.contact_shape_params:
        for key in shape_params.keys():
            if key not in shape_keys:
                shape_keys.append(key)

    contacts_table = ContactsTable(
        name="contacts_table",
        description="Contacts table for probeinterface",
    )
    for index in np.arange(probe.get_contact_count()):
        kwargs = dict(
            contact_position=contact_positions[index],
            contact_plane_axes=contact_plane_axes[index],
            contact_id=contact_ids[index],
            contact_shape=contacts_arr["contact_shapes"][index],
        )
        for key in shape_keys:
            kwargs[key] = contacts_arr[key][index]
        if probe.shank_ids is not None and len(probe.shank_ids) > 0:
            kwargs["shank_id"] = probe.shank_ids[index]
        contacts_table.add_row(kwargs)

    model_name = probe.model_name
    name = model_name if model_name is not None else f"ProbeModel{model_index}"
    # core DeviceModel requires manufacturer and rejects None for the optional
    # identity fields, so fill a sentinel manufacturer and pass the rest only when set.
    kwargs = dict(
        name=name,
        manufacturer=probe.manufacturer if probe.manufacturer is not None else _UNKNOWN_MANUFACTURER,
        ndim=probe.ndim,
        unit=unit_map[probe.si_units],
        contacts_table=contacts_table,
    )
    if model_name is not None:
        kwargs["model_number"] = model_name
        kwargs["description"] = model_name
    if probe.probe_planar_contour is not None:
        kwargs["planar_contour"] = probe.probe_planar_contour
    return ProbeModel(**kwargs)


def _probe_to_device(probe: Probe, probe_model, instance_index: int):
    from pynwb import get_class

    Probe = get_class("Probe", "ndx-probeinterface")

    annotations = probe.annotations.copy()
    for key in _PROMOTED_ANNOTATIONS:
        annotations.pop(key, None)

    name = probe.name if probe.name is not None else f"Probe{instance_index}"
    return Probe(
        name=name,
        serial_number=probe.serial_number,
        model=probe_model,
        annotations=json.dumps(annotations),
    )


def _get_contacts_table(probe_model):
    """The generated field name for the ContactsTable child."""
    for attribute in ("contacts_table", "ContactsTable", "contact_table"):
        table = getattr(probe_model, attribute, None)
        if table is not None:
            return table
    raise AttributeError("ProbeModel has no contacts table")
