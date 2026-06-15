from __future__ import annotations

import argparse
import math
import os
import random
from pathlib import Path

import neat
import pygame


SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
BASE_DIR = Path(__file__).resolve().parent
ASSET_DIR = BASE_DIR / "assets"

ROAD_COLOR = pygame.Color(255, 255, 255, 255)
ROAD_THRESHOLD = (20, 20, 20, 255)
RADAR_RANGE = 250

screen: pygame.Surface
clock: pygame.time.Clock
track: pygame.Surface
road_mask: pygame.mask.Mask
car_image: pygame.Surface

cars: list[pygame.sprite.GroupSingle]
ge: list[neat.DefaultGenome]
nets: list[neat.nn.FeedForwardNetwork]
pop: neat.Population

frame_limit = 800
draw_radars = True
history: list[dict[str, float]] = []


GOALS = [
    pygame.Rect(1170, 400, 80, 80),
    pygame.Rect(1174, 200, 80, 80),
    pygame.Rect(1000, 85, 80, 80),
    pygame.Rect(750, 120, 80, 80),
    pygame.Rect(520, 215, 80, 80),
    pygame.Rect(280, 120, 80, 80),
    pygame.Rect(60, 200, 80, 80),
    pygame.Rect(18, 400, 80, 80),
    pygame.Rect(125, 560, 80, 80),
    pygame.Rect(380, 425, 80, 80),
    pygame.Rect(750, 530, 80, 80),
]


def scale_image(image: pygame.Surface, factor: float) -> pygame.Surface:
    width = int(image.get_width() * factor)
    height = int(image.get_height() * factor)
    return pygame.transform.scale(image, (width, height))


def load_image(name: str) -> pygame.Surface:
    path = ASSET_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing asset: {path}")
    return pygame.image.load(str(path)).convert_alpha()


def is_on_road(point: tuple[int, int]) -> bool:
    x, y = point
    if not (0 <= x < SCREEN_WIDTH and 0 <= y < SCREEN_HEIGHT):
        return False
    return road_mask.get_at((x, y)) == 1


class Car(pygame.sprite.Sprite):
    def __init__(self) -> None:
        super().__init__()
        self.alpha_image = car_image.copy()
        self.image = self.alpha_image.copy()
        self.rect = self.image.get_rect()

        self.speed = 0.0
        self.max_speed = 24
        self.acceleration = 2
        self.direction = 0
        self.rotation_speed = 8
        self.angle = -35

        self.x = 1100 + random.randint(-20, 20)
        self.y = 590 + random.randint(-40, 13)
        self.rect.x = int(self.x)
        self.rect.y = int(self.y)

        self.alive = True
        self.radars: list[list[int]] = []
        self.start_radar = False
        self.prev_goal = GOALS[-1]
        self.next_goal = GOALS[0]
        self.score = 0

    def rotate(self) -> None:
        if self.direction == 1:
            self.angle += self.rotation_speed
        elif self.direction == -1:
            self.angle -= self.rotation_speed

    def move_forward(self) -> None:
        self.speed = min(self.speed + self.acceleration, self.max_speed)
        radians = math.radians(self.angle)
        self.y -= math.cos(radians) * self.speed
        self.x -= math.sin(radians) * self.speed

    def scan_radar(self, relative_angle: int) -> None:
        if not self.start_radar:
            return

        length = 0
        x, y = self.rect.center
        radar_angle = self.angle + 90 + relative_angle

        while is_on_road((x, y)) and length < RADAR_RANGE:
            length += 1
            x = int(self.rect.centerx + math.cos(math.radians(radar_angle)) * length)
            y = int(self.rect.centery - math.sin(math.radians(radar_angle)) * length)

        if draw_radars:
            pygame.draw.line(screen, (255, 0, 0), self.rect.center, (x, y), 1)
            pygame.draw.circle(screen, (255, 255, 0), (x, y), 2)

        distance = int(math.hypot(self.rect.centerx - x, self.rect.centery - y))
        self.radars.append([relative_angle, distance])

    def sensor_data(self) -> list[int]:
        values = {45: 0, 0: 0, -45: 0}
        for angle, distance in self.radars:
            values[angle] = distance
        return [values[45], values[0], values[-45]]

    def update_goal_progress(self) -> None:
        if not self.rect.colliderect(self.next_goal):
            return

        current_index = GOALS.index(self.next_goal)
        self.prev_goal = self.next_goal
        self.next_goal = GOALS[(current_index + 1) % len(GOALS)]
        self.score += 1

    def sample_collision_points(self) -> list[tuple[int, int]]:
        radius = max(4, min(self.rect.width, self.rect.height) // 3)
        center = self.rect.center
        return [
            center,
            (center[0] + radius, center[1]),
            (center[0] - radius, center[1]),
            (center[0], center[1] + radius),
            (center[0], center[1] - radius),
        ]

    def update(self) -> None:
        self.radars.clear()
        self.rotate()
        self.move_forward()

        rotated = pygame.transform.rotate(self.alpha_image, self.angle)
        self.rect = rotated.get_rect(center=self.image.get_rect(center=(self.x, self.y)).center)
        self.image = rotated

        self.scan_radar(45)
        self.scan_radar(0)
        self.scan_radar(-45)
        self.update_goal_progress()
        self.alive = all(is_on_road(point) for point in self.sample_collision_points())


def remove_car(index: int) -> None:
    cars.pop(index)
    ge.pop(index)
    nets.pop(index)


def init_pygame(headless: bool) -> None:
    global screen, clock, track, road_mask, car_image

    if headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    pygame.init()
    pygame.display.set_caption("neatRace")
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    track = load_image("Track1.png")
    mask_source = load_image("fondo1.png")
    road_mask = pygame.mask.from_threshold(mask_source, ROAD_COLOR, ROAD_THRESHOLD)
    car_image = scale_image(load_image("Car_3_01.png"), 0.04)


def eval_genomes(genomes, config) -> None:
    global cars, ge, nets

    cars = []
    ge = []
    nets = []
    current_generation = pop.generation

    for _, genome in genomes:
        genome.fitness = 0.0
        ge.append(genome)
        nets.append(neat.nn.FeedForwardNetwork.create(genome, config))
        car = pygame.sprite.GroupSingle(Car())
        car.sprite.start_radar = True
        cars.append(car)

    max_fitness = 0.0
    max_score = 0

    for frame in range(1, frame_limit + 1):
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        if not cars:
            break

        screen.blit(track, (0, 0))

        for i, car in enumerate(cars):
            output = nets[i].activate(car.sprite.sensor_data())
            if output[0] > 0.6:
                car.sprite.direction = 1
            elif output[1] > 0.6:
                car.sprite.direction = -1
            else:
                car.sprite.direction = 0

            car.sprite.update()
            car.draw(screen)
            ge[i].fitness = car.sprite.score * 100 + frame * 0.01
            max_fitness = max(max_fitness, ge[i].fitness)
            max_score = max(max_score, car.sprite.score)

        pygame.display.flip()

        for i in reversed(range(len(cars))):
            if not cars[i].sprite.alive:
                remove_car(i)

    history.append(
        {
            "generation": float(current_generation),
            "max_fitness": float(max_fitness),
            "max_score": float(max_score),
            "survivors": float(len(cars)),
        }
    )

    print(
        f"Generation {current_generation}: "
        f"max_fitness={max_fitness:.2f}, max_score={max_score}, survivors={len(cars)}"
    )


def plot_history(output_path: Path | None = None) -> None:
    if not history:
        return

    import matplotlib.pyplot as plt

    generations = [row["generation"] for row in history]
    fitness = [row["max_fitness"] for row in history]

    plt.plot(generations, fitness, color="#d38d2c")
    plt.xlabel("Generation")
    plt.ylabel("Max fitness")
    plt.title("neatRace training progress")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"Saved plot to {output_path}")
    else:
        plt.show()


def run(config_path: Path, generations: int, should_plot: bool, plot_file: Path | None) -> neat.DefaultGenome:
    global pop

    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(config_path),
    )

    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(neat.StatisticsReporter())

    winner = pop.run(eval_genomes, generations)
    print(f"Best genome fitness: {winner.fitness:.2f}")

    if should_plot:
        plot_history(plot_file)

    return winner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a NEAT agent to drive around the neatRace track.")
    parser.add_argument("--config", type=Path, default=BASE_DIR / "config.txt", help="Path to a neat-python config file.")
    parser.add_argument("--generations", type=int, default=50, help="Number of NEAT generations to train.")
    parser.add_argument("--frame-limit", type=int, default=800, help="Maximum frames per generation.")
    parser.add_argument("--headless", action="store_true", help="Run without opening a visible pygame window.")
    parser.add_argument("--no-radars", action="store_true", help="Hide radar debug lines.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducible starts.")
    parser.add_argument("--plot", action="store_true", help="Plot max fitness after training.")
    parser.add_argument("--plot-file", type=Path, default=None, help="Save the plot to a file instead of opening a window.")
    return parser.parse_args()


def main() -> None:
    global frame_limit, draw_radars

    args = parse_args()
    if args.generations < 1:
        raise ValueError("--generations must be at least 1")
    if args.frame_limit < 1:
        raise ValueError("--frame-limit must be at least 1")

    frame_limit = args.frame_limit
    draw_radars = not args.no_radars

    if args.seed is not None:
        random.seed(args.seed)

    init_pygame(args.headless)
    try:
        run(args.config, args.generations, args.plot, args.plot_file)
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
