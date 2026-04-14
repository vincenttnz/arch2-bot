import math


class CombatAI:

    def distance(self, a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def choose_target(self, player, enemies):
        if not enemies:
            return None

        # closest enemy
        enemies = sorted(enemies, key=lambda e: self.distance(player, e))
        return enemies[0]

    def dodge_vector(self, player, projectiles):
        dx, dy = 0, 0

        for p in projectiles:
            dist = self.distance(player, p)

            if dist < 150:
                dx += player[0] - p[0]
                dy += player[1] - p[1]

        return dx, dy

    def decide(self, frame, enemies, projectiles):
        h, w = frame.shape[:2]
        player = (w // 2, h // 2)

        target = self.choose_target(player, enemies)

        dx, dy = 0, 0

        # Move toward enemy
        if target:
            dx += target[0] - player[0]
            dy += target[1] - player[1]

        # Dodge projectiles
        ddx, ddy = self.dodge_vector(player, projectiles)

        dx += ddx * 2
        dy += ddy * 2

        # Normalize into direction
        if abs(dx) > abs(dy):
            return "right" if dx > 0 else "left"
        else:
            return "down" if dy > 0 else "up"
