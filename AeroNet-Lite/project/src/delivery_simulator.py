"""
delivery_simulator.py  -  Module 4
Clean 20-step simulation: exactly ONE logical action per step group.

Step 1  : Grid init + CSP validation
Step 2  : Hubs listed
Step 3  : Fleet created
Step 4  : Deliveries generated
Step 5  : Deliveries assigned to drones
Step 6  : Routes confirmed, drones dispatched
Steps 7-10 : Drones fly (movement only, no new assignments)
Step 11 : No-fly zone activated (always on an active route)
Steps 12-13: Rerouting check + drones continue flying
Step 14 : Drones continue flying
Step 15 : Demand forecast loaded
Step 16 : Remaining deliveries assigned (post-forecast)
Step 17 : Drones continue flying
Step 18 : Anomaly injected
Step 19 : Emergency return / drones continue
Step 20 : Final summary
"""

import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from grid_model import Cell, build_grid, get_hubs, manhattan
from astar_planner import plan_delivery_route, astar

random.seed(42)

# ── Dataclasses ────────────────────────────────────────────────────────────────
@dataclass
class Drone:
    drone_id: str
    drone_type: str
    position: Tuple[int, int]
    hub: Tuple[int, int]
    payload_kg: float
    range_cells: int
    battery: float = 100.0
    status: str = "Idle"
    current_delivery: Optional[int] = None
    route: List[Tuple[int, int]] = field(default_factory=list)
    route_index: int = 0
    anomaly: Optional[str] = None
    completed_route: List[Tuple[int, int]] = field(default_factory=list)

@dataclass
class Delivery:
    delivery_id: int
    pickup: Tuple[int, int]
    dropoff: Tuple[int, int]
    weight_kg: float
    status: str = "Pending"
    assigned_drone: Optional[str] = None

# ── Event log ──────────────────────────────────────────────────────────────────
event_log = []

def log(step: int, msg: str):
    e = f"Step {step:02d}: {msg}"
    event_log.append(e)
    print(e)

# ── Helpers ────────────────────────────────────────────────────────────────────

def create_drones(fleet, hubs):
    drones, did, hc = [], 1, 0
    for _ in range(fleet["light_drones"]):
        h = hubs[hc % len(hubs)]
        drones.append(Drone(f"D{did}", "Light", h, h, 2.0, 12))
        did += 1; hc += 1
    for _ in range(fleet["heavy_drones"]):
        h = hubs[hc % len(hubs)]
        drones.append(Drone(f"D{did}", "Heavy", h, h, 5.0, 20))
        did += 1; hc += 1
    return drones


def generate_deliveries(grid, count=4):
    res = [(r,c) for r in range(10) for c in range(10)
           if grid[r][c].zone == "Residential" and not grid[r][c].is_hub]
    com = [(r,c) for r in range(10) for c in range(10)
           if grid[r][c].zone in ("Commercial","Open Field") and not grid[r][c].is_hub]
    random.shuffle(res); random.shuffle(com)
    out = []
    for i in range(min(count, len(res), len(com))):
        out.append(Delivery(i+1, com[i], res[i], round(random.uniform(0.5,4.5),1)))
    return out


def assign_deliveries(drones, deliveries, grid, step):
    """Assign ALL pending deliveries to idle drones. One log line per assignment."""
    idle = [d for d in drones if d.status == "Idle"]
    assigned = 0
    for delivery in deliveries:
        if delivery.status != "Pending" or not idle:
            continue
        eligible = [d for d in idle if d.payload_kg >= delivery.weight_kg]
        if not eligible:
            continue
        drone = min(eligible, key=lambda d: manhattan(d.position, delivery.pickup))
        res = plan_delivery_route(drone.hub, delivery.pickup, delivery.dropoff, grid)
        if res["success"]:
            drone.route = res["path"]
            drone.route_index = 0
            drone.status = "EnRoute"
            drone.current_delivery = delivery.delivery_id
            delivery.status = "Assigned"
            delivery.assigned_drone = drone.drone_id
            idle.remove(drone)
            assigned += 1
            log(step, f"Delivery {delivery.delivery_id} -> Drone {drone.drone_id} "
                      f"| pickup={delivery.pickup} dropoff={delivery.dropoff} "
                      f"| route={len(res['path'])} cells.")
    return assigned


def advance_drones(drones, deliveries, grid, step, cells_per_step=3):
    """
    Move every active drone up to `cells_per_step` cells.
    One log line per drone that completes its delivery.
    Movement itself is silent (no per-cell spam).
    """
    for drone in drones:
        if drone.status not in ("EnRoute", "Rerouting"):
            continue
        for _ in range(cells_per_step):
            if drone.route_index >= len(drone.route) - 1:
                # Delivery complete
                drone.position = drone.route[-1]
                drone.completed_route = list(drone.route)
                drone.status = "Idle"
                if drone.current_delivery is not None:
                    for d in deliveries:
                        if d.delivery_id == drone.current_delivery:
                            d.status = "Completed"
                            log(step, f"Drone {drone.drone_id} completed "
                                      f"Delivery {drone.current_delivery} "
                                      f"at {drone.position}.")
                    drone.current_delivery = None
                drone.route = []
                drone.route_index = 0
                break
            else:
                drone.route_index += 1
                drone.position = drone.route[drone.route_index]
                drone.battery = max(0.0, drone.battery - random.uniform(0.2, 0.5))


def pick_no_fly_cell(drones, deliveries, grid):
    """
    Pick a no-fly cell guaranteed to be on an active drone's remaining route.
    Avoids hubs, pickups, dropoffs, and cells within 2 steps of the end.
    """
    protected = set(get_hubs(grid))
    for d in deliveries:
        protected.add(d.pickup)
        protected.add(d.dropoff)

    best_drone, best_cell = None, None
    best_len = 0

    for drone in drones:
        if drone.status not in ("EnRoute", "Rerouting"):
            continue
        remaining = drone.route[drone.route_index + 1:]
        # Ignore last 2 cells (near destination) and first 1 cell (just left)
        candidates = remaining[4:-2]  # skip first 4 cells so drone is never near the no-fly cell
        valid = [(r, c) for (r, c) in candidates
                 if (r, c) not in protected
                 and not grid[r][c].is_hub
                 and not grid[r][c].is_charging
                 and not grid[r][c].no_fly]
        if valid and len(remaining) > best_len:
            best_len = len(remaining)
            best_drone = drone
            # pick cell at 1/3 of remaining to give room for rerouting
            best_cell = valid[len(valid) // 2]  # middle of route = maximum rerouting buffer

    return best_cell, (best_drone.drone_id if best_drone else None)


def reroute_affected_drones(drones, deliveries, grid, step):
    """Reroute any drone whose remaining path crosses a no-fly cell."""
    for drone in drones:
        if drone.status not in ("EnRoute", "Rerouting"):
            continue
        remaining = drone.route[drone.route_index:]
        conflict = next(((r,c) for (r,c) in remaining if grid[r][c].no_fly), None)
        if conflict is None:
            continue

        delivery = next((d for d in deliveries
                         if d.delivery_id == drone.current_delivery), None)
        if delivery is None:
            continue

        log(step, f"Drone {drone.drone_id} path blocked at {conflict}. Running A* reroute.")
        reroute = plan_delivery_route(drone.position, delivery.pickup,
                                       delivery.dropoff, grid)
        if reroute["success"]:
            drone.route = reroute["path"]
            drone.route_index = 0
            drone.status = "Rerouting"
            log(step, f"Drone {drone.drone_id} rerouted successfully "
                      f"| new route={len(reroute['path'])} cells.")
        else:
            drone.status = "Failed"
            delivery.status = "Delayed"
            log(step, f"Drone {drone.drone_id} cannot reroute safely. "
                      f"Delivery {delivery.delivery_id} DELAYED.")


def inject_anomaly(drones, step):
    """Inject one anomaly into a random active drone."""
    active = [d for d in drones if d.status in ("EnRoute", "Rerouting")]
    if not active:
        active = [d for d in drones if d.status == "Idle"]
    if not active:
        return
    drone = random.choice(active)
    atype = random.choice(["Battery Anomaly", "Route Anomaly", "Sensor Spike"])
    drone.anomaly = atype
    drop = random.uniform(15, 35)
    drone.battery = max(0.0, drone.battery - drop)
    log(step, f"{atype} on Drone {drone.drone_id} "
              f"| battery dropped to {drone.battery:.1f}%.")
    if drone.battery < 20 and drone.status == "EnRoute":
        drone.status = "ReturnHub"
        log(step, f"Drone {drone.drone_id} status -> ReturnHub (critical battery).")


# ── Snapshot (required by Streamlit visualization) ────────────────────────────

def take_snapshot(step, drones, deliveries, grid):
    return {
        "step": step,
        "drones": [
            {
                "id":       d.drone_id,
                "type":     d.drone_type,
                "pos":      d.position,
                "hub":      d.hub,
                "status":   d.status,
                "battery":  round(d.battery, 1),
                "anomaly":  d.anomaly,
                "delivery": d.current_delivery,
                "trail":    list(d.route[:d.route_index + 1]) if d.route else
                            list(d.completed_route),
                "remaining": list(d.route[d.route_index + 1:]) if d.route else [],
            }
            for d in drones
        ],
        "no_fly_cells": [(r, c) for r in range(10)
                         for c in range(10) if grid[r][c].no_fly],
        "deliveries": [
            {"id": d.delivery_id, "status": d.status,
             "pickup": d.pickup, "dropoff": d.dropoff,
             "assigned": d.assigned_drone}
            for d in deliveries
        ],
        "summary": {
            "completed": sum(1 for d in deliveries if d.status == "Completed"),
            "delayed":   sum(1 for d in deliveries if d.status == "Delayed"),
            "failed":    sum(1 for d in deliveries if d.status == "Failed"),
            "active":    sum(1 for d in drones
                             if d.status in ("EnRoute", "Rerouting")),
        },
    }


# ── Main simulation ────────────────────────────────────────────────────────────

def run_simulation(grid, fleet, demand_forecast=None):
    event_log.clear()
    snapshots = []
    hubs = get_hubs(grid)

    # Step 1: Grid initialization
    log(1, "Grid initialized. 10x10 city layout loaded.")
    log(1, f"Grid: 10x10 cells | Hubs: {len(hubs)} | Budget: ${fleet['budget']}.")
    snapshots.append(take_snapshot(1, [], [], grid))

    # Step 2: CSP Layout Validation
    from layout_validator import run_validator
    report = run_validator(grid)
    if report["valid"]:
        log(2, "CSP Layout Validation PASSED. All 4 rules satisfied (R1-R4).")
    else:
        log(2, f"CSP Layout Validation: {len(report['failed'])} rule(s) failed. Continuing.")
    snapshots.append(take_snapshot(2, [], [], grid))

    # Step 3: Fleet selected by GA
    drones = create_drones(fleet, hubs)
    log(3, f"Fleet selected: {fleet['light_drones']} Light + {fleet['heavy_drones']} Heavy "
          f"| {fleet['total_drones']} drones | cost=${fleet['total_cost']} "
          f"| fitness={fleet['fitness_score']}.")
    snapshots.append(take_snapshot(3, drones, [], grid))

    # Step 4
    deliveries = generate_deliveries(grid, count=5)
    log(4, f"Generated {len(deliveries)} deliveries.")
    snapshots.append(take_snapshot(4, drones, deliveries, grid))

    # Step 5
    assign_deliveries(drones, deliveries, grid, 5)
    snapshots.append(take_snapshot(5, drones, deliveries, grid))

    # Step 6
    active = sum(1 for d in drones if d.status == "EnRoute")
    log(6, f"{active} drone(s) dispatched. Routes confirmed via A* search.")
    snapshots.append(take_snapshot(6, drones, deliveries, grid))

    # Steps 7-10: fly only
    for step in range(7, 11):
        advance_drones(drones, deliveries, grid, step, cells_per_step=3)
        active = sum(1 for d in drones if d.status in ("EnRoute", "Rerouting"))
        done   = sum(1 for d in deliveries if d.status == "Completed")
        log(step, f"Drones flying. Active={active} | Completed={done}.")
        snapshots.append(take_snapshot(step, drones, deliveries, grid))

    # Step 11: no-fly zone activated + immediately reroute so drone is never shown on no-fly
    nf_cell, nf_drone = pick_no_fly_cell(drones, deliveries, grid)
    if nf_cell:
        grid[nf_cell[0]][nf_cell[1]].no_fly = True
        log(11, f"NO-FLY ZONE activated at {nf_cell} (intersects Drone {nf_drone} route).")
        # Immediately reroute any affected drone so snapshot is clean
        reroute_affected_drones(drones, deliveries, grid, 11)
    else:
        nf_cell = (3, 4)
        grid[3][4].no_fly = True
        log(11, f"NO-FLY ZONE activated at {nf_cell} (precautionary).")

    # Step 12: reroute
    reroute_affected_drones(drones, deliveries, grid, 12)
    advance_drones(drones, deliveries, grid, 12, cells_per_step=3)
    done = sum(1 for d in deliveries if d.status == "Completed")
    log(12, f"Rerouting complete. Drones continuing. Completed={done}.")
    snapshots.append(take_snapshot(12, drones, deliveries, grid))

    # Step 13
    advance_drones(drones, deliveries, grid, 13, cells_per_step=3)
    done = sum(1 for d in deliveries if d.status == "Completed")
    log(13, f"Drones flying. Completed={done}.")
    snapshots.append(take_snapshot(13, drones, deliveries, grid))

    # Step 14
    advance_drones(drones, deliveries, grid, 14, cells_per_step=3)
    done = sum(1 for d in deliveries if d.status == "Completed")
    log(14, f"Drones flying. Completed={done}.")
    snapshots.append(take_snapshot(14, drones, deliveries, grid))

    # Step 15: demand forecast
    if demand_forecast:
        avg = demand_forecast.get("avg_demand", 0)
        log(15, f"Demand forecast integrated. Predicted avg demand = {avg:.2f} units (Random Forest).")
    else:
        log(15, "Demand forecast unavailable. Skipping.")
    snapshots.append(take_snapshot(15, drones, deliveries, grid))

    # Step 16: assign remaining
    n2 = assign_deliveries(drones, deliveries, grid, 16)
    if n2 == 0:
        log(16, "All deliveries already assigned or no idle drones available.")
    snapshots.append(take_snapshot(16, drones, deliveries, grid))

    # Step 17
    advance_drones(drones, deliveries, grid, 17, cells_per_step=3)
    done = sum(1 for d in deliveries if d.status == "Completed")
    log(17, f"Drones flying. Completed={done}.")
    snapshots.append(take_snapshot(17, drones, deliveries, grid))

    # Step 18: anomaly
    inject_anomaly(drones, 18)
    snapshots.append(take_snapshot(18, drones, deliveries, grid))

    # Step 19: Reroute or force return to hub if anomaly detected
    returned = 0
    for drone in drones:
        if drone.status == "ReturnHub":
            ret = astar(drone.position, drone.hub, grid)
            if ret["success"]:
                drone.route = ret["path"]
                drone.route_index = 0
                returned += 1
                log(19, f"Drone {drone.drone_id} forced return to hub "
                        f"{drone.hub} due to anomaly (route={len(ret['path'])} cells).")
    if returned == 0:
        # Check for any rerouting still needed from no-fly zone
        reroute_affected_drones(drones, deliveries, grid, 19)
        active = sum(1 for d in drones if d.status in ("EnRoute","Rerouting"))
        log(19, f"No emergency returns needed. {active} drone(s) continuing normally.")
    advance_drones(drones, deliveries, grid, 19, cells_per_step=3)
    done = sum(1 for d in deliveries if d.status == "Completed")
    log(19, f"Step 19 complete. Completed so far: {done}/{len(deliveries)}.")
    snapshots.append(take_snapshot(19, drones, deliveries, grid))

    # Step 20: finalize
    for d in deliveries:
        if d.status in ("Pending", "Assigned"):
            d.status = "Delayed"
            log(20, f"Delivery {d.delivery_id} marked Delayed (not finished in time).")

    completed = sum(1 for d in deliveries if d.status == "Completed")
    delayed   = sum(1 for d in deliveries if d.status == "Delayed")
    failed    = sum(1 for d in deliveries if d.status == "Failed")
    pending   = sum(1 for d in deliveries if d.status == "Pending")
    log(20, f"Simulation complete | Completed={completed} | Delayed={delayed} | Failed={failed}.")
    snapshots.append(take_snapshot(20, drones, deliveries, grid))

    return {
        "completed": completed, "delayed": delayed,
        "failed": failed,       "pending": pending,
        "total": len(deliveries),
        "drones": drones,
        "deliveries": deliveries,
        "event_log": list(event_log),
        "no_fly_cell": nf_cell,
        "snapshots": snapshots,
    }