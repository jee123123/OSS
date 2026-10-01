# This example is not working in Spyder directly (F5 or Run)
# Please type '!python turtle_runaway.py' on IPython console in your Spyder.
import tkinter as tk
import turtle, random, time, math

class RunawayGame:
    def __init__(self, canvas, runner, chaser, catch_radius=50):
        self.canvas = canvas
        self.runner = runner
        self.chaser = chaser
        self.catch_radius2 = catch_radius**2

        # Initialize 'runner' and 'chaser'
        self.runner.shape('turtle')
        self.runner.color('blue')
        self.runner.penup()

        self.chaser.shape('turtle')
        self.chaser.color('red')
        self.chaser.penup()

        # Instantiate another turtle for drawing
        self.drawer = turtle.RawTurtle(canvas)
        self.drawer.hideturtle()
        self.drawer.penup()

    def is_catched(self):
        p = self.runner.pos()
        q = self.chaser.pos()
        dx, dy = p[0] - q[0], p[1] - q[1]
        return dx**2 + dy**2 < self.catch_radius2

    def start(self, init_dist=400, ai_timer_msec=100):
        self.runner.setpos((-init_dist / 2, 0))
        self.runner.setheading(0)
        self.chaser.setpos((+init_dist / 2, 0))
        self.chaser.setheading(180)

        self.time_limit = 30
        self.start_time = time.time()
        self.score = 0
        self.is_game_over = False

        # TODO) You can do something here and follows.
        self.ai_timer_msec = ai_timer_msec
        self.canvas.ontimer(self.step, self.ai_timer_msec)

    def step(self):
        self.runner.run_ai(self.chaser.pos(), self.chaser.heading())
        self.chaser.run_ai(self.runner.pos(), self.runner.heading())

        # TODO) You can do something here and follows.
        elapsed_time = time.time() - self.start_time
        time_left = max(0, int(self.time_limit - elapsed_time))
        
        
        is_catched = self.is_catched()
        if is_catched:
            self.score += 10
            self.runner.setpos(random.randint(-200, 200), random.randint(-200, 200))
        
        self.drawer.undo()
        self.drawer.penup()
        self.drawer.setpos(-300, 300)
        self.drawer.write(f'Time Left: {time_left}s | Score: {self.score} | Is catched? {is_catched}')

        if time_left <= 0:
            self.is_game_over = True
            self.drawer.setpos(-100, 0)
            self.drawer.write(f'GAME OVER!\nFinal Score: {self.score}', font=('Arial', 16, 'bold'))
            return
        
        # Note) The following line should be the last of this function to keep the game playing
        self.canvas.ontimer(self.step, self.ai_timer_msec)

class ManualMover(turtle.RawTurtle):
    def __init__(self, canvas, step_move=10, step_turn=10):
        super().__init__(canvas)
        self.step_move = step_move
        self.step_turn = step_turn

        # Register event handlers
        canvas.onkeypress(lambda: self.forward(self.step_move), 'Up')
        canvas.onkeypress(lambda: self.backward(self.step_move), 'Down')
        canvas.onkeypress(lambda: self.left(self.step_turn), 'Left')
        canvas.onkeypress(lambda: self.right(self.step_turn), 'Right')
        canvas.listen()

    def move_forward(self):
        self.forward(self.step_move)
        if abs(self.xcor()) > self.boundary or abs(self.ycor()) > self.boundary:
            self.backward(self.step_move)

    def move_backward(self):
        self.backward(self.step_move)
        if abs(self.xcor()) > self.boundary or abs(self.ycor()) > self.boundary:
            self.forward(self.step_move)

    def run_ai(self, opp_pos, opp_heading):
        pass

class RandomMover(turtle.RawTurtle):
    def __init__(self, canvas, step_move=10, step_turn=10):
        super().__init__(canvas)
        self.step_move = step_move
        self.step_turn = step_turn

    def run_ai(self, opp_pos, opp_heading):
        mode = random.randint(0, 2)
        if mode == 0:
            self.forward(self.step_move)
        elif mode == 1:
            self.left(self.step_turn)
        elif mode == 2:
            self.right(self.step_turn)

class SmartMover(RandomMover):
    def __init__(self, canvas, step_move=10, step_turn=10, boundary=300, safe_distance=150):
        super().__init__(canvas, step_move, step_turn)
        self.boundary = boundary
        self.safe_distance = safe_distance

    def run_ai(self, opp_pos, opp_heading):
        x, y = self.pos()
        if abs(x) > self.boundary or abs(y) > self.boundary:
            center_angle = math.degrees(math.atan2(-y, -x))
            self.setheading(center_angle)
            self.forward(self.step_move)
            return

        dx = x - opp_pos[0]
        dy = y - opp_pos[1]
        dist = math.hypot(dx, dy)

        if dist > self.safe_distance:
            super().run_ai(opp_pos, opp_heading)
        else:
            escape_angle = math.degrees(math.atan2(dy, dx))
            self.setheading(escape_angle)
            self.forward(self.step_move)

if __name__ == '__main__':
    # Use 'TurtleScreen' instead of 'Screen' to prevent an exception from the singleton 'Screen'
    root = tk.Tk()
    canvas = tk.Canvas(root, width=700, height=700)
    canvas.pack()
    screen = turtle.TurtleScreen(canvas)

    # TODO) Change the follows to your turtle if necessary
    runner = RandomMover(screen)
    chaser = ManualMover(screen)

    game = RunawayGame(screen, runner, chaser)
    game.start()
    screen.mainloop()
