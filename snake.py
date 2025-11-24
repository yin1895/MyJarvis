import pygame
import sys
import random

# 初始化 Pygame
pygame.init()

# 定义颜色
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# 屏幕尺寸
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600

# 蛇块大小
BLOCK_SIZE = 20

# 创建屏幕
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('贪吃蛇')

# 时钟
clock = pygame.time.Clock()

# 字体
font = pygame.font.SysFont(None, 35)

# 蛇的初始位置和方向
def init_snake():
    return [[SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2], [SCREEN_WIDTH // 2 - BLOCK_SIZE, SCREEN_HEIGHT // 2], [SCREEN_WIDTH // 2 - 2 * BLOCK_SIZE, SCREEN_HEIGHT // 2]]

snake = init_snake()
snake_direction = 'RIGHT'

# 食物位置
def random_food():
    x = round(random.randrange(0, SCREEN_WIDTH - BLOCK_SIZE) / BLOCK_SIZE) * BLOCK_SIZE
    y = round(random.randrange(0, SCREEN_HEIGHT - BLOCK_SIZE) / BLOCK_SIZE) * BLOCK_SIZE
    return [x, y]

food = random_food()

# 分数
score = 0

# 游戏结束函数
def game_over():
    screen.fill(BLACK)
    text = font.render('游戏结束! 分数: ' + str(score), True, WHITE)
    screen.blit(text, [SCREEN_WIDTH // 2 - 150, SCREEN_HEIGHT // 2])
    pygame.display.update()
    pygame.time.wait(2000)
    pygame.quit()
    sys.exit()

# 主游戏循环
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP and snake_direction != 'DOWN':
                snake_direction = 'UP'
            elif event.key == pygame.K_DOWN and snake_direction != 'UP':
                snake_direction = 'DOWN'
            elif event.key == pygame.K_LEFT and snake_direction != 'RIGHT':
                snake_direction = 'LEFT'
            elif event.key == pygame.K_RIGHT and snake_direction != 'LEFT':
                snake_direction = 'RIGHT'

    # 移动蛇头
    head = snake[0].copy()
    if snake_direction == 'UP':
        head[1] -= BLOCK_SIZE
    elif snake_direction == 'DOWN':
        head[1] += BLOCK_SIZE
    elif snake_direction == 'LEFT':
        head[0] -= BLOCK_SIZE
    elif snake_direction == 'RIGHT':
        head[0] += BLOCK_SIZE

    snake.insert(0, head)

    # 检查吃食物
    if snake[0] == food:
        score += 10
        food = random_food()
        # 确保食物不在蛇身上
        while food in snake:
            food = random_food()
    else:
        snake.pop()

    # 检查碰撞
    if (snake[0][0] >= SCREEN_WIDTH or snake[0][0] < 0 or
        snake[0][1] >= SCREEN_HEIGHT or snake[0][1] < 0 or
        snake[0] in snake[1:]):  # 撞墙或自己
        game_over()

    # 绘制屏幕
    screen.fill(BLACK)

    # 绘制蛇
    for block in snake:
        pygame.draw.rect(screen, GREEN, pygame.Rect(block[0], block[1], BLOCK_SIZE, BLOCK_SIZE))

    # 绘制食物
    pygame.draw.rect(screen, RED, pygame.Rect(food[0], food[1], BLOCK_SIZE, BLOCK_SIZE))

    # 绘制分数
    score_text = font.render('分数: ' + str(score), True, WHITE)
    screen.blit(score_text, [0, 0])

    pygame.display.update()
    clock.tick(10)  # 控制速度

pygame.quit()
sys.exit()