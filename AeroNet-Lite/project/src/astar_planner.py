"""
astar_planner.py
Module 3 - Delivery Path Planning using A* Search.
Plans hub -> pickup -> dropoff -> hub routes on the 10x10 grid.
"""

import heapq
from typing import List, Optional, Tuple
from grid_model import Cell, manhattan, get_hubs


def astar(start: Tuple[int, int], goal: Tuple[int, int], grid: List[List[Cell]]) -> dict:
    """
    A* search from start to goal on the 10x10 grid.
    - Cost: 1.0 per normal move, 0.8 for Commercial cells (corridors).
    - Blocked: cells where no_fly = True.
    - Heuristic: Manhattan distance (admissible).
    Returns: {path, cost, success, message}
    """
    if grid[goal[0]][goal[1]].no_fly:
        return {"path": [], "cost": 0.0, "success": False,
                "message": f"Goal {goal} is a no-fly zone."}

    # Priority queue: (f_cost, g_cost, position, parent_path)
    open_heap = []
    heapq.heappush(open_heap, (0 + manhattan(start, goal), 0.0, start, [start]))

    visited = {}  # position -> best g_cost seen

    while open_heap:
        f, g, current, path = heapq.heappop(open_heap)

        if current == goal:
            return {"path": path, "cost": round(g, 2), "success": True,
                    "message": f"Path found: {len(path)} steps, cost={round(g,2)}."}

        if current in visited and visited[current] <= g:
            continue
        visited[current] = g

        r, c = current
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 10 and 0 <= nc < 10:
                neighbor = grid[nr][nc]
                if neighbor.no_fly:
                    continue
                # Step cost: 0.8 for Commercial (corridor discount), else 1.0
                step_cost = 0.8 if neighbor.zone == "Commercial" else 1.0
                new_g = g + step_cost
                if (nr, nc) not in visited or visited[(nr, nc)] > new_g:
                    h = manhattan((nr, nc), goal)
                    heapq.heappush(open_heap, (new_g + h, new_g, (nr, nc), path + [(nr, nc)]))

    return {"path": [], "cost": 0.0, "success": False,
            "message": f"No path from {start} to {goal}. Destination unreachable."}


def plan_delivery_route(
    hub: Tuple[int, int],
    pickup: Tuple[int, int],
    dropoff: Tuple[int, int],
    grid: List[List[Cell]]
) -> dict:
    """
    Full delivery route: hub -> pickup -> dropoff -> hub.
    Returns combined path and total cost.
    """
    seg1 = astar(hub, pickup, grid)
    if not seg1["success"]:
        return {"success": False, "message": f"Hub to pickup failed: {seg1['message']}",
                "path": [], "total_cost": 0.0}

    seg2 = astar(pickup, dropoff, grid)
    if not seg2["success"]:
        return {"success": False, "message": f"Pickup to dropoff failed: {seg2['message']}",
                "path": [], "total_cost": 0.0}

    seg3 = astar(dropoff, hub, grid)
    if not seg3["success"]:
        return {"success": False, "message": f"Dropoff to hub failed: {seg3['message']}",
                "path": [], "total_cost": 0.0}

    # Combine paths (avoid duplicate waypoints at junctions)
    full_path = seg1["path"] + seg2["path"][1:] + seg3["path"][1:]
    total_cost = seg1["cost"] + seg2["cost"] + seg3["cost"]

    return {
        "success": True,
        "message": "Full route planned successfully.",
        "hub": hub, "pickup": pickup, "dropoff": dropoff,
        "path": full_path,
        "total_cost": round(total_cost, 2),
        "segments": {
            "hub_to_pickup": seg1,
            "pickup_to_dropoff": seg2,
            "dropoff_to_hub": seg3
        }
    }


def assign_nearest_hub(drone_pos: Tuple[int, int], grid: List[List[Cell]]) -> Optional[Tuple[int, int]]:
    """Return the nearest hub to a given position."""
    hubs = get_hubs(grid)
    if not hubs:
        return None
    return min(hubs, key=lambda h: manhattan(drone_pos, h))


if __name__ == "__main__":
    from grid_model import build_grid
    grid = build_grid()
    hubs = get_hubs(grid)
    hub = hubs[0]
    pickup = (6, 4)
    dropoff = (8, 8)
    print(f"Planning route: hub={hub} -> pickup={pickup} -> dropoff={dropoff}")
    result = plan_delivery_route(hub, pickup, dropoff, grid)
    print(f"Success: {result['success']}")
    print(f"Total Cost: {result['total_cost']}")
    print(f"Path Length: {len(result['path'])} steps")
    print(f"Path: {result['path']}")
