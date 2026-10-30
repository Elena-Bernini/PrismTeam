# Dataset and implemented schema

## Scope

The current implementation is wired to RPLAN through the Graph2Plan converted data files. ResPlan and the future Maticad dataset are documented as forward-compatible targets but are not loaded by this codebase yet.

## Room type IDs

The layout JSON uses the official Graph2Plan/RPLAN room categories observed in the converted data:

| ID | Name |
|---:|---|
| 0 | LivingRoom |
| 1 | MasterRoom |
| 2 | Kitchen |
| 3 | Bathroom |
| 4 | DiningRoom |
| 5 | ChildRoom |
| 6 | StudyRoom |
| 7 | SecondRoom |
| 8 | GuestRoom |
| 9 | Balcony |
| 10 | Entrance |
| 11 | Storage |
| 12 | Wall-in |

Reserved labels sometimes documented in Graph2Plan/RPLAN tooling are `External`, `ExteriorWall`, `FrontDoor`, `InteriorWall`, and `InteriorDoor`, but these should not normally appear as room boxes in `data_train_converted.pkl`.

## Layout JSON fields

Each room contains `id`, `room_type_id`, `room_type_name`, `zone`, `bbox`, `width`, `height`, `area_pixels`, `area_ratio`, `area_sqm`, `centroid`, `grid_location_5x5`, and when available `polygon`.

At layout level, preprocessing now writes `scale_meters_per_pixel`, `total_room_area_pixels`, `total_room_area_ratio`, and `total_area_sqm`.

## Area caveat

RPLAN/Graph2Plan samples are normalized to a 256x256 coordinate canvas. The code therefore treats `area_pixels` and `area_ratio` as reliable, but leaves `area_sqm` and `total_area_sqm` as `null` unless a verified `scale_meters_per_pixel` is supplied by config. No RPLAN-wide meters-per-pixel constant is fabricated.

## Semantic zones

Zones are loaded from `config/room_zones.yaml`, not hardcoded in stage code. Default mapping:

- Private Zone: MasterRoom, ChildRoom, SecondRoom, GuestRoom, Bathroom, Wall-in
- Social Zone: LivingRoom, DiningRoom, Kitchen, StudyRoom
- External Zone: Balcony
- Service/Circulation Zone: Entrance, Storage

## Door/window feasibility investigation

The currently consumed Graph2Plan converted files expose room boxes, room polygons, boundaries, draw order, and relation edges. The available archive/file names and inspected MATLAB sample fields did not expose reliable door or window geometry. Therefore this project does **not** invent door or window positions in visualizations. Entrance rooms remain visible as semantic room labels, and exact door/window rendering is deferred until a dataset with verified annotations is integrated, such as ResPlan or Maticad CAD data.

## Graph JSON

Each graph node corresponds to a room and carries room type, zone, area, bbox, centroid, and 5x5 grid location. Edges preserve Graph2Plan relation IDs/names and describe the source room relative to the target room.
