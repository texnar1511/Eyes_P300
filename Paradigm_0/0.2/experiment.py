import pygame, math, random, os
from pylsl import StreamInfo, StreamOutlet, local_clock
import numpy as np
from datetime import datetime
from scipy.stats import rv_continuous
from distributions import Absolute

class Experiment:
    
    
    def __init__(self, data_config):
        
        self.data_config = data_config
        self.sentence = self.data_config['sentence']
        self.circle_color = self.data_config['circle_color']
        self.circle_radius = self.data_config['circle_radius']
        
    
    def fit(self):
        
        self.preinit()
        self.init()
        self.lslinit()
        
        for target_letter in self.sentence:
            self.postinit()
            self.perform(target_letter)
        
        
        self.end_exp()
        
    
    def preinit(self):
        
        x = self.data_config['window_x']
        y = self.data_config['window_y']
        os.environ['SDL_VIDEO_WINDOW_POS'] = f'{x},{y}'
        os.environ['SDL_VIDEO_MINIMIZE_ON_FOCUS_LOSS'] = '0'
        
    
    def init(self):
        
        pygame.init()
        flags = pygame.FULLSCREEN
        self.screen = pygame.display.set_mode(size = (0, 0), flags = flags)

        self.width = self.screen.get_width()
        self.height = self.screen.get_height()
        
        self.n_cols = self.data_config['num_cols']
        self.n_rows = self.data_config['num_rows']

        self.alphabet = self.data_config['alphabet'].replace('\n', '')

        self.w, self.h = self.width // self.n_cols, self.height // self.n_rows

        self.sec_in_msec = 1e-3

        self.t0_mean = self.data_config['t0_mean']

        self.t1_a = self.data_config['t1_a']
        self.t1_b = self.data_config['t1_b']

        self.t2_a = self.data_config['t2_a']
        self.t2_b = self.data_config['t2_b']
        
        self.delay = self.data_config['delay']
        self.prepare_rest = self.data_config['prepare_rest']
        
        self.path = f'logs\logs_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.txt'
        os.makedirs(os.path.dirname(self.path), exist_ok = True)
        
        self.f = open(self.path, 'w', encoding = 'utf-8')
        
    
    def set_t1(self):
        
        return np.random.uniform(self.t1_a, self.t1_b)

    
    def set_t0(self):
        
        return self.t0_mean

    
    def set_t2(self):
        
        return np.random.uniform(self.t2_a, self.t2_b)
        
    
    def speed_func(self, t, t0 = 1, t1 = 0.5, t2 = 0.5):
    
        if t0 <= 0: raise ValueError('freq must be positive')

        if t1 < 0: raise ValueError('t1 must be non negative')

        if t2 < 0: raise ValueError('t2 must be non negative')

        if t1 / self.sec_in_msec <= t <= (t1 + t0) / self.sec_in_msec:

            return (t - t1 / self.sec_in_msec) * math.tau * self.sec_in_msec / t0
            
        elif 0 <= t < t1 / self.sec_in_msec or (t1 + t0) / self.sec_in_msec < t <= (t1 + t0 + t2) / self.sec_in_msec:

            return 0

        else:

            raise ValueError(f't must be between 0 and {(t1 + t0 + t2) / self.sec_in_msec}')
    
    def noise_support_var0(self, t):
        
        return np.sin(t) ** 2
    
    def noise_support_var1(self, t):
        
        return np.sin(t / 2) ** 2       
        
    def noise(self, t, sigma):
        
        return self.noise_support_var0(t) * np.random.normal(0, sigma)
    
    def apogee_var0(self):
        
        return np.random.uniform((0, 0), (self.width, self.height), 2)
    
    def apogee_var1(self, x0, y0):
        
        return Absolute(a = 0, b = self.width).rvs(x0), Absolute(a = 0, b = self.height).rvs(y0)


    def lslinit(self):
        
        self.info = StreamInfo(name = 'annotations', type = 'Events', channel_count = 1, nominal_srate = 0, channel_format = 'string', source_id = 'my_marker_stream')

        self.outlet = StreamOutlet(self.info)
        
    
    def postinit(self):

        self.clock = pygame.time.Clock()

        self.FPS = self.data_config['FPS']

        self.font_name = self.data_config['font_name']
        self.font_size = self.data_config['font_size']

        self.font = pygame.font.SysFont(name = self.font_name, size = self.font_size, bold = True)

        self.cells = {}

        self.amplitude_x_scale = self.data_config['amplitude_x_scale']

        self.amplitude_y_scale = self.data_config['amplitude_y_scale']
        
        self.letter_foreground = (0, 0, 0)
        
        self.background = (255, 255, 255)

        for id, char in enumerate(self.alphabet):
            letter = self.font.render(char, True, self.letter_foreground)
            
            j, i = id // self.n_cols, id % self.n_cols
            x0, y0 = i * self.w + self.w / 2, j * self.h + self.h / 2
            
            #p_ellipse = self.apogee_var1(x0, y0)
            #print(p_ellipse)
            
            p_ellipse = self.apogee_var0()
            
            self.cells[char] = {
                'id': id,
                'letter': letter,
                'amplitude_x': self.w * self.amplitude_x_scale,
                'amplitude_y': self.h * self.amplitude_y_scale,
                't0': self.set_t0(),
                't1': self.set_t1(),
                't2': self.set_t2(),
                'start_t': 0,
                'start_marker': False,
                'end_marker': False,
                'circle_color': np.random.randint(0, 256, 3),
                'b_ellipse': 10,
                'p_ellipse': p_ellipse
                }

        self.running = True

        self.start_experiment = False
        
        self.prev_t = -self.delay
        
    def perform(self, target_letter):
        
        self.f.write(f'target letter {target_letter}\n')

        while self.running:
    
            dt = self.clock.tick(self.FPS)
    
            for event in pygame.event.get():
        
                if event.type == pygame.QUIT:
                    self.end_exp()
            
                if event.type == pygame.KEYDOWN:
            
                    if event.key == pygame.K_ESCAPE:
                        self.end_exp()


                    if event.key == pygame.K_SPACE:

                        self.start_experiment = not self.start_experiment
                        
                        if self.start_experiment:

                            t0 = pygame.time.get_ticks()
                        
                            self.prev_t = -self.delay
                        
                        else:
                            
                            self.running = False
    
            self.screen.fill(self.background)
    
            if self.start_experiment:
                t = pygame.time.get_ticks() - t0
                if t - self.prev_t >= self.delay:
                    self.prev_t = t
                    self.outlet.push_sample([str(t)], local_clock())
                    self.f.write(f'sync {t}\n')

            
    
            for char, params in self.cells.items():

                j, i = params['id'] // self.n_cols, params['id'] % self.n_cols

                if self.start_experiment:

                    t = pygame.time.get_ticks() - t0

                    try:
                        speed = self.speed_func(t - params['start_t'], params['t0'], params['t1'], params['t2'])

                        if t >= params['start_t'] + params['t1'] / self.sec_in_msec and not params['start_marker']:

                            self.f.write(f'letter {char} start {t}\n')

                            params['start_marker'] = True

                        if t >= params['start_t'] + (params['t1'] + params['t0']) / self.sec_in_msec and not params['end_marker']:

                            params['end_marker'] = True

                    except ValueError:
                    
                        params['start_t'] = t
                        params['t1'] = self.set_t1()
                        params['t2'] = self.set_t2()

                        params['start_marker'] = False
                        params['end_marker'] = False

                        speed = self.speed_func(t - params['start_t'], params['t0'], params['t1'], params['t2'])



                    #dx = params['amplitude_x'] * speed if self.data_config['x_move'] else 0
                    #dy = params['amplitude_y'] * speed if self.data_config['y_move'] else 0
                    
                    x0, y0 = i * self.w + self.w / 2, j * self.h + self.h / 2
                    x1, y1 = params['p_ellipse']
                    
                    xc, yc = (x0 + x1) / 2, (y0 + y1) / 2
                    
                    ax, ay = (x0 - x1) / 2, (y0 - y1) / 2
                    
                    a_norm = np.sqrt(ax ** 2 + ay ** 2)
                    
                    b = params['b_ellipse']
                    
                    bx, by = -b * ay / a_norm, b * ax / a_norm
                    
                    dx = xc + ax * np.cos(speed) + bx * np.sin(speed)
                    dy = yc + ay * np.cos(speed) + by * np.sin(speed)
                    
                    sigma = 1
                    
                    #noise = self.noise(t - params['start_t'], sigma)
                    
                    #dx += noise
                    #dy += noise
                    
                    self.screen.blit(params['letter'], (x0 - params['letter'].get_width() / 2, y0 - params['letter'].get_height() / 2))
                    
                    pygame.draw.circle(self.screen, params['circle_color'], (dx, dy), self.circle_radius, 0)


                else:
                    dx = 0
                    dy = 0
                    
                    if char == target_letter:
                        self.screen.blit(params['letter'], (i * self.w + self.w / 2 - params['letter'].get_width() / 2 + dx, j * self.h + self.h / 2 - params['letter'].get_height() / 2 - dy))


            pygame.display.flip()


    def end_exp(self):
        
        self.f.close()
        pygame.quit()