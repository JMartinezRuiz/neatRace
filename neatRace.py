import pygame, sys
from ultils import scale_image
import math
import neat
import os
import random
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np



# pygame setting
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()
pygame.init()

# Goals rects
goal1_rect = pygame.Rect(1170, 400, 80, 80)
goal2_rect = pygame.Rect(1174, 200, 80, 80)
goal3_rect = pygame.Rect(1000, 85, 80, 80)
goal4_rect = pygame.Rect(750, 120, 80, 80)
goal5_rect = pygame.Rect(520, 215, 80, 80)
goal6_rect = pygame.Rect(280, 120, 80, 80)
goal7_rect = pygame.Rect(60, 200, 80, 80)
goal8_rect = pygame.Rect(18, 400, 80, 80)
goal9_rect = pygame.Rect(125, 560, 80, 80)
goal10_rect = pygame.Rect(380, 425, 80, 80)
goal11_rect = pygame.Rect(750, 530, 80, 80)

goals_rects = [goal1_rect, goal2_rect, goal3_rect, goal4_rect, goal5_rect, goal6_rect, goal7_rect, goal8_rect,
               goal9_rect, goal10_rect, goal11_rect]


# Classes
class Car(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = scale_image(pygame.image.load('assets/Car_3_01.png'), 0.04)
        self.alpha_image = scale_image(pygame.image.load('assets/Car_3_01.png'), 0.04)
        self.rect = self.image.get_rect()

        self.speed = 0
        self.max_speed = 24  # test with 12
        self.acceleration = 2

        self.direction = 0

        self.rotation_speed = 8  # test with 4
        self.angle = -35

        self.x = 1100 + random.randint(-20, 20)
        self.y = 590 + random.randint(-40, 13)
        #self.x = 1100
        #self.y = 590

        self.rect.x = self.x
        self.rect.y = self.y

        self.radar1_rect = None
        self.start_radar = False

        self.alive = True

        self.moved = False
        self.rotated = False

        self.radars = []

        self.prev_goal = goals_rects[10]
        self.next_goal = goals_rects[0]
        self.score = 0

    def rotate(self):
        if self.direction == 1:
            self.angle += self.rotation_speed
        if self.direction == -1:
            self.angle -= self.rotation_speed

    def move_forward(self):
        self.speed = min(self.speed + self.acceleration, self.max_speed)
        self.move()

    def reduce_speed(self):
        self.speed = max(self.speed - self.acceleration / 1.2, 0)
        self.move()

    def move_backward(self):
        self.speed = max(self.speed - self.acceleration, 0)
        self.move()

    def move(self):
        radians = math.radians(self.angle)
        vertical = math.cos(radians) * self.speed
        horizontal = math.sin(radians) * self.speed

        self.y -= vertical
        self.x -= horizontal
        self.update()

    def bounce(self):
        self.speed = -self.speed / 2
        self.move()

    def radar(self, r_angle):

        if self.start_radar:

            length = 0  # longitud de la línea en píxeles
            radians = math.radians(self.angle + 90 + r_angle)
            x = int(self.rect.center[0])
            y = int(self.rect.center[1])

            if not (0 <= self.rect.center[0] < SCREEN_WIDTH and 0 <= self.rect.center[1] < SCREEN_HEIGHT):
                return
            while not screen.get_at((x, y)) == pygame.Color(255, 255, 255,
                                                            255) and length < 250 and 0 <= x < SCREEN_WIDTH - 10 and 0 <= y < SCREEN_HEIGHT - 10:
                length += 1
                x = int(self.rect.center[0] + math.cos(math.radians(self.angle + 90 + r_angle)) * length)
                y = int(self.rect.center[1] - math.sin(math.radians(self.angle + 90 + r_angle)) * length)

            ##Draw Radar##
            pygame.draw.line(screen, (255, 0, 0, 255), self.rect.center, (x, y), 1)
            pygame.draw.circle(screen, (255, 255, 0, 255), (x, y), 2)

            dist = int(math.sqrt(math.pow(self.rect.center[0] - x, 2) + math.pow(self.rect.center[1] - y, 2)))

            self.radars.append([r_angle, dist])

    def data(self):
        iinput = [0, 0, 0]
        for i, radar in enumerate(self.radars):
            if radar[0] == 45:
                iinput[0] = int(radar[1])
            elif radar[0] == 0:
                iinput[1] = int(radar[1])
            elif radar[0] == -45:
                iinput[2] = int(radar[1])
        return iinput

    def goals(self):
        if self.rect.colliderect(self.next_goal):
            index_prev = int(goals_rects.index(self.prev_goal))
            print(f'Collision, Index is: {index_prev}')
            print(f'Prev goal is: {self.prev_goal}')
            print(f'Next goal is = {self.next_goal}')

            if self.prev_goal == goals_rects[10]:
                self.prev_goal = goals_rects[0]
                self.next_goal = goals_rects[1]
                print("+ Fit")
                self.score += 1

            elif self.next_goal == goals_rects[9]:
                self.prev_goal = goals_rects[9]
                self.next_goal = goals_rects[10]
                print("+ Fit")
                self.score += 1

            elif self.next_goal == goals_rects[10]:
                self.prev_goal = goals_rects[10]
                self.next_goal = goals_rects[0]
                #print("+ Fit")
                self.score += 1

            elif self.next_goal == goals_rects[index_prev + 1]:
                self.prev_goal = self.next_goal
                self.next_goal = goals_rects[index_prev + 2]
                print("+ Fit")
                self.score += 1

    def update(self):

        self.radars.clear()

        if not self.moved:
            self.moved = True
            self.move_forward()

        if not self.rotated:
            self.rotated = True
            self.rotate()

        rotated_image = pygame.transform.rotate(self.alpha_image, self.angle)
        new_rect = rotated_image.get_rect(center=self.image.get_rect(center=(self.x, self.y)).center)
        self.image = rotated_image
        self.rect = new_rect
        self.radar(45)
        self.radar(0)
        self.radar(-45)
        #print(self.radars)
        self.goals()
        # self.data()
        self.moved = False
        self.rotated = False


def remove(index):
    cars.pop(index)
    ge.pop(index)
    nets.pop(index)


class Obstacle(pygame.sprite.Sprite):
    def __init__(self, start_pos):
        super().__init__()
        self.image = scale_image(pygame.image.load('assets/fondo1.png'), 1)
        self.rect = self.image.get_rect()


def plotting(data1, data2):
    # TEMP, Only data1

    data1 = data1.drop(df.index[-1])
    data2 = data2.drop(df.index[-1])

    print(data1)
    print(data2)

    xx = data1['generation']
    yy = data1['max_fitness']

    xz = data2['generation']
    z1 = data2['car1']
    z2 = data2['car2']
    z3 = data2['car3']
    z4 = data2['car4']
    z5 = data2['car5']

    plt.plot(xx, yy, color='#fbdba3')

    plt.xticks(range(df.iloc[0]['generation'], df.iloc[-1]['generation'] + 1))

    plt.xlabel('Generation')
    plt.ylabel('Max Fitness')
    # plt.legend()
    plt.show()


box = Obstacle((0, 0))
obstacle_group = pygame.sprite.GroupSingle(box)

# Load Track
track = pygame.image.load('assets/Track1.png')
track_rect = track.get_rect()

# PANDAS // TEMP
df = pd.DataFrame(columns=["generation", "total_score", "max_fitness"])
df2 = pd.DataFrame(columns=["generation", "car1", "car2", "car3", "car4", "car5"])


### Main Loop ###
def eval_genomes(genomes, config):
    global cars, ge, nets
    global df
    global df2

    cars = []
    ge = []
    nets = []

    print(df)
    print(df2)
    current_generation = pop.generation
    print(current_generation)
    if pop.best_genome is not None and pop.best_genome.fitness is not None:
        print("El fitness del ganador es:", pop.best_genome.fitness)
        df.iloc[-1, df.columns.get_loc('max_fitness')] = pop.best_genome.fitness

    new_row = pd.DataFrame({'generation': [current_generation], 'total_score': [0], 'max_fitness': [0]})
    df = pd.concat([df, new_row], ignore_index=True)

    print("PRINTING DF2")
    new_row_2 = pd.DataFrame(
        {'generation': [current_generation], 'car1': [0], 'car2': [0], 'car3': [0], 'car4': [0], 'car5': [0]})
    df2 = pd.concat([df2, new_row_2], ignore_index=True)

    print(df2)

    # End with df shape
    if df.shape[0] >= 46:
        print("Done")
        plotting(df, df2)
        exit()

    for genome_id, genome in genomes:
        cars.append(pygame.sprite.GroupSingle(Car()))
        ge.append(genome)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        genome.fitness = 0

    for car in cars:
        car.update()  # fix jump issue

    # Data
    total_score = 0
    limit = 0
    run = True
    while run:
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        limit += 1

        screen.blit(track, (0, 0))
        # Break generation after limit (in frames)
        if limit >= 800:  # Time limit
            for car in cars:
                total_score += car.sprite.score
                df.loc[df['generation'] == current_generation, 'total_score'] += total_score
                for genome_id, genome in genomes:
                    print(f"Genome {genome_id} fitness: {genome.fitness}")
                i = 0
                for genome_id, genome in genomes:
                    df2.iloc[current_generation, i + 1] = genome.fitness
                    i += 1
            break
        # Break generation if no cars left
        if len(cars) == 0:
            i = 0
            for genome_id, genome in genomes:
                df2.iloc[current_generation, i + 1] = genome.fitness
                i += 1
            break

        # Add fitness to each genome as self.score
        for i, car in enumerate(cars):
            ge[i].fitness = car.sprite.score
            # if car is dead add score to pandas and remove
            if not car.sprite.alive:
                car_score = car.sprite.score
                df.loc[df['generation'] == current_generation, 'total_score'] += car_score
                remove(i)

        for i, car in enumerate(cars):
            output = nets[i].activate(car.sprite.data())
            if output[0] > 0.6:
                car.sprite.direction = 1
            if output[1] > 0.6:
                car.sprite.direction = -1
            if output[0] <= 0.6 and output[1] <= 0.6:
                car.sprite.direction = 0

        obstacle_group.draw(screen)


        # DRAW
        for car in cars:
            car.draw(screen)
            car.update()

        pygame.display.update()

        # Collide
        for i, car in enumerate(cars):
            # Check collide
            if pygame.sprite.spritecollide(car.sprite, obstacle_group, False, pygame.sprite.collide_mask):
                print(f"Car {i} has collided with the obstacle!")
                car.sprite.alive = False
        # Radars
        for car in cars:
            car.sprite.start_radar = True


# NEAT
def run(config_path):
    global pop
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    pop.run(eval_genomes, 50)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    run(config_path)
