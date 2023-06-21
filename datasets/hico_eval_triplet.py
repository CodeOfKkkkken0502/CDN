import numpy as np
from collections import defaultdict
import os, cv2, json

hico_text_label = {(4, 4): 'a photo of a person boarding an airplane',
                   (17, 4): 'a photo of a person directing an airplane',
                   (25, 4): 'a photo of a person exiting an airplane',
                   (30, 4): 'a photo of a person flying an airplane',
                   (41, 4): 'a photo of a person inspecting an airplane',
                   (52, 4): 'a photo of a person loading an airplane',
                   (76, 4): 'a photo of a person riding an airplane',
                   (87, 4): 'a photo of a person sitting on an airplane',
                   (111, 4): 'a photo of a person washing an airplane',
                   (57, 4): 'a photo of a person and an airplane', (8, 1): 'a photo of a person carrying a bicycle',
                   (36, 1): 'a photo of a person holding a bicycle',
                   (41, 1): 'a photo of a person inspecting a bicycle',
                   (43, 1): 'a photo of a person jumping a bicycle',
                   (37, 1): 'a photo of a person hopping on a bicycle',
                   (62, 1): 'a photo of a person parking a bicycle',
                   (71, 1): 'a photo of a person pushing a bicycle',
                   (75, 1): 'a photo of a person repairing a bicycle',
                   (76, 1): 'a photo of a person riding a bicycle',
                   (87, 1): 'a photo of a person sitting on a bicycle',
                   (98, 1): 'a photo of a person straddling a bicycle',
                   (110, 1): 'a photo of a person walking a bicycle',
                   (111, 1): 'a photo of a person washing a bicycle', (57, 1): 'a photo of a person and a bicycle',
                   (10, 14): 'a photo of a person chasing a bird', (26, 14): 'a photo of a person feeding a bird',
                   (36, 14): 'a photo of a person holding a bird', (65, 14): 'a photo of a person petting a bird',
                   (74, 14): 'a photo of a person releasing a bird',
                   (112, 14): 'a photo of a person watching a bird', (57, 14): 'a photo of a person and a bird',
                   (4, 8): 'a photo of a person boarding a boat', (21, 8): 'a photo of a person driving a boat',
                   (25, 8): 'a photo of a person exiting a boat', (41, 8): 'a photo of a person inspecting a boat',
                   (43, 8): 'a photo of a person jumping a boat', (47, 8): 'a photo of a person launching a boat',
                   (75, 8): 'a photo of a person repairing a boat', (76, 8): 'a photo of a person riding a boat',
                   (77, 8): 'a photo of a person rowing a boat', (79, 8): 'a photo of a person sailing a boat',
                   (87, 8): 'a photo of a person sitting on a boat',
                   (93, 8): 'a photo of a person standing on a boat', (105, 8): 'a photo of a person tying a boat',
                   (111, 8): 'a photo of a person washing a boat', (57, 8): 'a photo of a person and a boat',
                   (8, 39): 'a photo of a person carrying a bottle',
                   (20, 39): 'a photo of a person drinking with a bottle',
                   (36, 39): 'a photo of a person holding a bottle',
                   (41, 39): 'a photo of a person inspecting a bottle',
                   (48, 39): 'a photo of a person licking a bottle',
                   (58, 39): 'a photo of a person opening a bottle',
                   (69, 39): 'a photo of a person pouring a bottle', (57, 39): 'a photo of a person and a bottle',
                   (4, 5): 'a photo of a person boarding a bus', (17, 5): 'a photo of a person directing a bus',
                   (21, 5): 'a photo of a person driving a bus', (25, 5): 'a photo of a person exiting a bus',
                   (41, 5): 'a photo of a person inspecting a bus', (52, 5): 'a photo of a person loading a bus',
                   (76, 5): 'a photo of a person riding a bus', (87, 5): 'a photo of a person sitting on a bus',
                   (111, 5): 'a photo of a person washing a bus', (113, 5): 'a photo of a person waving a bus',
                   (57, 5): 'a photo of a person and a bus', (4, 2): 'a photo of a person boarding a car',
                   (17, 2): 'a photo of a person directing a car', (21, 2): 'a photo of a person driving a car',
                   (38, 2): 'a photo of a person hosing a car', (41, 2): 'a photo of a person inspecting a car',
                   (43, 2): 'a photo of a person jumping a car', (52, 2): 'a photo of a person loading a car',
                   (62, 2): 'a photo of a person parking a car', (76, 2): 'a photo of a person riding a car',
                   (111, 2): 'a photo of a person washing a car', (57, 2): 'a photo of a person and a car',
                   (22, 15): 'a photo of a person drying a cat', (26, 15): 'a photo of a person feeding a cat',
                   (36, 15): 'a photo of a person holding a cat', (39, 15): 'a photo of a person hugging a cat',
                   (45, 15): 'a photo of a person kissing a cat', (65, 15): 'a photo of a person petting a cat',
                   (80, 15): 'a photo of a person scratching a cat', (111, 15): 'a photo of a person washing a cat',
                   (10, 15): 'a photo of a person chasing a cat', (57, 15): 'a photo of a person and a cat',
                   (8, 56): 'a photo of a person carrying a chair', (36, 56): 'a photo of a person holding a chair',
                   (49, 56): 'a photo of a person lying on a chair',
                   (87, 56): 'a photo of a person sitting on a chair',
                   (93, 56): 'a photo of a person standing on a chair', (57, 56): 'a photo of a person and a chair',
                   (8, 57): 'a photo of a person carrying a couch',
                   (49, 57): 'a photo of a person lying on a couch',
                   (87, 57): 'a photo of a person sitting on a couch', (57, 57): 'a photo of a person and a couch',
                   (26, 19): 'a photo of a person feeding a cow', (34, 19): 'a photo of a person herding a cow',
                   (36, 19): 'a photo of a person holding a cow', (39, 19): 'a photo of a person hugging a cow',
                   (45, 19): 'a photo of a person kissing a cow', (46, 19): 'a photo of a person lassoing a cow',
                   (55, 19): 'a photo of a person milking a cow', (65, 19): 'a photo of a person petting a cow',
                   (76, 19): 'a photo of a person riding a cow', (110, 19): 'a photo of a person walking a cow',
                   (57, 19): 'a photo of a person and a cow',
                   (12, 60): 'a photo of a person cleaning a dining table',
                   (24, 60): 'a photo of a person eating at a dining table',
                   (86, 60): 'a photo of a person sitting at a dining table',
                   (57, 60): 'a photo of a person and a dining table',
                   (8, 16): 'a photo of a person carrying a dog', (22, 16): 'a photo of a person drying a dog',
                   (26, 16): 'a photo of a person feeding a dog', (33, 16): 'a photo of a person grooming a dog',
                   (36, 16): 'a photo of a person holding a dog', (38, 16): 'a photo of a person hosing a dog',
                   (39, 16): 'a photo of a person hugging a dog', (41, 16): 'a photo of a person inspecting a dog',
                   (45, 16): 'a photo of a person kissing a dog', (65, 16): 'a photo of a person petting a dog',
                   (78, 16): 'a photo of a person running a dog', (80, 16): 'a photo of a person scratching a dog',
                   (98, 16): 'a photo of a person straddling a dog',
                   (107, 16): 'a photo of a person training a dog', (110, 16): 'a photo of a person walking a dog',
                   (111, 16): 'a photo of a person washing a dog', (10, 16): 'a photo of a person chasing a dog',
                   (57, 16): 'a photo of a person and a dog', (26, 17): 'a photo of a person feeding a horse',
                   (33, 17): 'a photo of a person grooming a horse',
                   (36, 17): 'a photo of a person holding a horse', (39, 17): 'a photo of a person hugging a horse',
                   (43, 17): 'a photo of a person jumping a horse', (45, 17): 'a photo of a person kissing a horse',
                   (52, 17): 'a photo of a person loading a horse',
                   (37, 17): 'a photo of a person hopping on a horse',
                   (65, 17): 'a photo of a person petting a horse', (72, 17): 'a photo of a person racing a horse',
                   (76, 17): 'a photo of a person riding a horse', (78, 17): 'a photo of a person running a horse',
                   (98, 17): 'a photo of a person straddling a horse',
                   (107, 17): 'a photo of a person training a horse',
                   (110, 17): 'a photo of a person walking a horse',
                   (111, 17): 'a photo of a person washing a horse', (57, 17): 'a photo of a person and a horse',
                   (36, 3): 'a photo of a person holding a motorcycle',
                   (41, 3): 'a photo of a person inspecting a motorcycle',
                   (43, 3): 'a photo of a person jumping a motorcycle',
                   (37, 3): 'a photo of a person hopping on a motorcycle',
                   (62, 3): 'a photo of a person parking a motorcycle',
                   (71, 3): 'a photo of a person pushing a motorcycle',
                   (72, 3): 'a photo of a person racing a motorcycle',
                   (76, 3): 'a photo of a person riding a motorcycle',
                   (87, 3): 'a photo of a person sitting on a motorcycle',
                   (98, 3): 'a photo of a person straddling a motorcycle',
                   (108, 3): 'a photo of a person turning a motorcycle',
                   (110, 3): 'a photo of a person walking a motorcycle',
                   (111, 3): 'a photo of a person washing a motorcycle',
                   (57, 3): 'a photo of a person and a motorcycle', (8, 0): 'a photo of a person carrying a person',
                   (31, 0): 'a photo of a person greeting a person',
                   (36, 0): 'a photo of a person holding a person', (39, 0): 'a photo of a person hugging a person',
                   (45, 0): 'a photo of a person kissing a person',
                   (92, 0): 'a photo of a person stabbing a person',
                   (100, 0): 'a photo of a person tagging a person',
                   (102, 0): 'a photo of a person teaching a person',
                   (48, 0): 'a photo of a person licking a person', (57, 0): 'a photo of a person and a person',
                   (8, 58): 'a photo of a person carrying a potted plant',
                   (36, 58): 'a photo of a person holding a potted plant',
                   (38, 58): 'a photo of a person hosing a potted plant',
                   (57, 58): 'a photo of a person and a potted plant',
                   (8, 18): 'a photo of a person carrying a sheep', (26, 18): 'a photo of a person feeding a sheep',
                   (34, 18): 'a photo of a person herding a sheep', (36, 18): 'a photo of a person holding a sheep',
                   (39, 18): 'a photo of a person hugging a sheep', (45, 18): 'a photo of a person kissing a sheep',
                   (65, 18): 'a photo of a person petting a sheep', (76, 18): 'a photo of a person riding a sheep',
                   (83, 18): 'a photo of a person shearing a sheep',
                   (110, 18): 'a photo of a person walking a sheep',
                   (111, 18): 'a photo of a person washing a sheep', (57, 18): 'a photo of a person and a sheep',
                   (4, 6): 'a photo of a person boarding a train', (21, 6): 'a photo of a person driving a train',
                   (25, 6): 'a photo of a person exiting a train', (52, 6): 'a photo of a person loading a train',
                   (76, 6): 'a photo of a person riding a train', (87, 6): 'a photo of a person sitting on a train',
                   (111, 6): 'a photo of a person washing a train', (57, 6): 'a photo of a person and a train',
                   (13, 62): 'a photo of a person controlling a tv', (75, 62): 'a photo of a person repairing a tv',
                   (112, 62): 'a photo of a person watching a tv', (57, 62): 'a photo of a person and a tv',
                   (7, 47): 'a photo of a person buying an apple', (15, 47): 'a photo of a person cutting an apple',
                   (23, 47): 'a photo of a person eating an apple',
                   (36, 47): 'a photo of a person holding an apple',
                   (41, 47): 'a photo of a person inspecting an apple',
                   (64, 47): 'a photo of a person peeling an apple',
                   (66, 47): 'a photo of a person picking an apple',
                   (89, 47): 'a photo of a person smelling an apple',
                   (111, 47): 'a photo of a person washing an apple', (57, 47): 'a photo of a person and an apple',
                   (8, 24): 'a photo of a person carrying a backpack',
                   (36, 24): 'a photo of a person holding a backpack',
                   (41, 24): 'a photo of a person inspecting a backpack',
                   (58, 24): 'a photo of a person opening a backpack',
                   (114, 24): 'a photo of a person wearing a backpack',
                   (57, 24): 'a photo of a person and a backpack', (7, 46): 'a photo of a person buying a banana',
                   (8, 46): 'a photo of a person carrying a banana',
                   (15, 46): 'a photo of a person cutting a banana',
                   (23, 46): 'a photo of a person eating a banana',
                   (36, 46): 'a photo of a person holding a banana',
                   (41, 46): 'a photo of a person inspecting a banana',
                   (64, 46): 'a photo of a person peeling a banana',
                   (66, 46): 'a photo of a person picking a banana',
                   (89, 46): 'a photo of a person smelling a banana', (57, 46): 'a photo of a person and a banana',
                   (5, 34): 'a photo of a person breaking a baseball bat',
                   (8, 34): 'a photo of a person carrying a baseball bat',
                   (36, 34): 'a photo of a person holding a baseball bat',
                   (84, 34): 'a photo of a person signing a baseball bat',
                   (99, 34): 'a photo of a person swinging a baseball bat',
                   (104, 34): 'a photo of a person throwing a baseball bat',
                   (115, 34): 'a photo of a person wielding a baseball bat',
                   (57, 34): 'a photo of a person and a baseball bat',
                   (36, 35): 'a photo of a person holding a baseball glove',
                   (114, 35): 'a photo of a person wearing a baseball glove',
                   (57, 35): 'a photo of a person and a baseball glove',
                   (26, 21): 'a photo of a person feeding a bear', (40, 21): 'a photo of a person hunting a bear',
                   (112, 21): 'a photo of a person watching a bear', (57, 21): 'a photo of a person and a bear',
                   (12, 59): 'a photo of a person cleaning a bed', (49, 59): 'a photo of a person lying on a bed',
                   (87, 59): 'a photo of a person sitting on a bed', (57, 59): 'a photo of a person and a bed',
                   (41, 13): 'a photo of a person inspecting a bench',
                   (49, 13): 'a photo of a person lying on a bench',
                   (87, 13): 'a photo of a person sitting on a bench', (57, 13): 'a photo of a person and a bench',
                   (8, 73): 'a photo of a person carrying a book', (36, 73): 'a photo of a person holding a book',
                   (58, 73): 'a photo of a person opening a book', (73, 73): 'a photo of a person reading a book',
                   (57, 73): 'a photo of a person and a book', (36, 45): 'a photo of a person holding a bowl',
                   (96, 45): 'a photo of a person stirring a bowl', (111, 45): 'a photo of a person washing a bowl',
                   (48, 45): 'a photo of a person licking a bowl', (57, 45): 'a photo of a person and a bowl',
                   (15, 50): 'a photo of a person cutting a broccoli',
                   (23, 50): 'a photo of a person eating a broccoli',
                   (36, 50): 'a photo of a person holding a broccoli',
                   (89, 50): 'a photo of a person smelling a broccoli',
                   (96, 50): 'a photo of a person stirring a broccoli',
                   (111, 50): 'a photo of a person washing a broccoli',
                   (57, 50): 'a photo of a person and a broccoli', (3, 55): 'a photo of a person blowing a cake',
                   (8, 55): 'a photo of a person carrying a cake', (15, 55): 'a photo of a person cutting a cake',
                   (23, 55): 'a photo of a person eating a cake', (36, 55): 'a photo of a person holding a cake',
                   (51, 55): 'a photo of a person lighting a cake', (54, 55): 'a photo of a person making a cake',
                   (67, 55): 'a photo of a person picking up a cake', (57, 55): 'a photo of a person and a cake',
                   (8, 51): 'a photo of a person carrying a carrot',
                   (14, 51): 'a photo of a person cooking a carrot',
                   (15, 51): 'a photo of a person cutting a carrot',
                   (23, 51): 'a photo of a person eating a carrot',
                   (36, 51): 'a photo of a person holding a carrot',
                   (64, 51): 'a photo of a person peeling a carrot',
                   (89, 51): 'a photo of a person smelling a carrot',
                   (96, 51): 'a photo of a person stirring a carrot',
                   (111, 51): 'a photo of a person washing a carrot', (57, 51): 'a photo of a person and a carrot',
                   (8, 67): 'a photo of a person carrying a cell phone',
                   (36, 67): 'a photo of a person holding a cell phone',
                   (73, 67): 'a photo of a person reading a cell phone',
                   (75, 67): 'a photo of a person repairing a cell phone',
                   (101, 67): 'a photo of a person talking on a cell phone',
                   (103, 67): 'a photo of a person texting on a cell phone',
                   (57, 67): 'a photo of a person and a cell phone',
                   (11, 74): 'a photo of a person checking a clock',
                   (36, 74): 'a photo of a person holding a clock',
                   (75, 74): 'a photo of a person repairing a clock',
                   (82, 74): 'a photo of a person setting a clock', (57, 74): 'a photo of a person and a clock',
                   (8, 41): 'a photo of a person carrying a cup',
                   (20, 41): 'a photo of a person drinking with a cup',
                   (36, 41): 'a photo of a person holding a cup', (41, 41): 'a photo of a person inspecting a cup',
                   (69, 41): 'a photo of a person pouring a cup', (85, 41): 'a photo of a person sipping a cup',
                   (89, 41): 'a photo of a person smelling a cup', (27, 41): 'a photo of a person filling a cup',
                   (111, 41): 'a photo of a person washing a cup', (57, 41): 'a photo of a person and a cup',
                   (7, 54): 'a photo of a person buying a donut', (8, 54): 'a photo of a person carrying a donut',
                   (23, 54): 'a photo of a person eating a donut', (36, 54): 'a photo of a person holding a donut',
                   (54, 54): 'a photo of a person making a donut',
                   (67, 54): 'a photo of a person picking up a donut',
                   (89, 54): 'a photo of a person smelling a donut', (57, 54): 'a photo of a person and a donut',
                   (26, 20): 'a photo of a person feeding an elephant',
                   (36, 20): 'a photo of a person holding an elephant',
                   (38, 20): 'a photo of a person hosing an elephant',
                   (39, 20): 'a photo of a person hugging an elephant',
                   (45, 20): 'a photo of a person kissing an elephant',
                   (37, 20): 'a photo of a person hopping on an elephant',
                   (65, 20): 'a photo of a person petting an elephant',
                   (76, 20): 'a photo of a person riding an elephant',
                   (110, 20): 'a photo of a person walking an elephant',
                   (111, 20): 'a photo of a person washing an elephant',
                   (112, 20): 'a photo of a person watching an elephant',
                   (57, 20): 'a photo of a person and an elephant',
                   (39, 10): 'a photo of a person hugging a fire hydrant',
                   (41, 10): 'a photo of a person inspecting a fire hydrant',
                   (58, 10): 'a photo of a person opening a fire hydrant',
                   (61, 10): 'a photo of a person painting a fire hydrant',
                   (57, 10): 'a photo of a person and a fire hydrant',
                   (36, 42): 'a photo of a person holding a fork', (50, 42): 'a photo of a person lifting a fork',
                   (95, 42): 'a photo of a person sticking a fork', (48, 42): 'a photo of a person licking a fork',
                   (111, 42): 'a photo of a person washing a fork', (57, 42): 'a photo of a person and a fork',
                   (2, 29): 'a photo of a person blocking a frisbee',
                   (9, 29): 'a photo of a person catching a frisbee',
                   (36, 29): 'a photo of a person holding a frisbee',
                   (90, 29): 'a photo of a person spinning a frisbee',
                   (104, 29): 'a photo of a person throwing a frisbee',
                   (57, 29): 'a photo of a person and a frisbee', (26, 23): 'a photo of a person feeding a giraffe',
                   (45, 23): 'a photo of a person kissing a giraffe',
                   (65, 23): 'a photo of a person petting a giraffe',
                   (76, 23): 'a photo of a person riding a giraffe',
                   (112, 23): 'a photo of a person watching a giraffe',
                   (57, 23): 'a photo of a person and a giraffe',
                   (36, 78): 'a photo of a person holding a hair drier',
                   (59, 78): 'a photo of a person operating a hair drier',
                   (75, 78): 'a photo of a person repairing a hair drier',
                   (57, 78): 'a photo of a person and a hair drier',
                   (8, 26): 'a photo of a person carrying a handbag',
                   (36, 26): 'a photo of a person holding a handbag',
                   (41, 26): 'a photo of a person inspecting a handbag',
                   (57, 26): 'a photo of a person and a handbag', (8, 52): 'a photo of a person carrying a hot dog',
                   (14, 52): 'a photo of a person cooking a hot dog',
                   (15, 52): 'a photo of a person cutting a hot dog',
                   (23, 52): 'a photo of a person eating a hot dog',
                   (36, 52): 'a photo of a person holding a hot dog',
                   (54, 52): 'a photo of a person making a hot dog', (57, 52): 'a photo of a person and a hot dog',
                   (8, 66): 'a photo of a person carrying a keyboard',
                   (12, 66): 'a photo of a person cleaning a keyboard',
                   (36, 66): 'a photo of a person holding a keyboard',
                   (109, 66): 'a photo of a person typing on a keyboard',
                   (57, 66): 'a photo of a person and a keyboard', (1, 33): 'a photo of a person assembling a kite',
                   (8, 33): 'a photo of a person carrying a kite', (30, 33): 'a photo of a person flying a kite',
                   (36, 33): 'a photo of a person holding a kite',
                   (41, 33): 'a photo of a person inspecting a kite',
                   (47, 33): 'a photo of a person launching a kite', (70, 33): 'a photo of a person pulling a kite',
                   (57, 33): 'a photo of a person and a kite', (16, 43): 'a photo of a person cutting with a knife',
                   (36, 43): 'a photo of a person holding a knife',
                   (95, 43): 'a photo of a person sticking a knife',
                   (111, 43): 'a photo of a person washing a knife',
                   (115, 43): 'a photo of a person wielding a knife',
                   (48, 43): 'a photo of a person licking a knife', (57, 43): 'a photo of a person and a knife',
                   (36, 63): 'a photo of a person holding a laptop',
                   (58, 63): 'a photo of a person opening a laptop',
                   (73, 63): 'a photo of a person reading a laptop',
                   (75, 63): 'a photo of a person repairing a laptop',
                   (109, 63): 'a photo of a person typing on a laptop',
                   (57, 63): 'a photo of a person and a laptop',
                   (12, 68): 'a photo of a person cleaning a microwave',
                   (58, 68): 'a photo of a person opening a microwave',
                   (59, 68): 'a photo of a person operating a microwave',
                   (57, 68): 'a photo of a person and a microwave',
                   (13, 64): 'a photo of a person controlling a mouse',
                   (36, 64): 'a photo of a person holding a mouse',
                   (75, 64): 'a photo of a person repairing a mouse', (57, 64): 'a photo of a person and a mouse',
                   (7, 49): 'a photo of a person buying an orange',
                   (15, 49): 'a photo of a person cutting an orange',
                   (23, 49): 'a photo of a person eating an orange',
                   (36, 49): 'a photo of a person holding an orange',
                   (41, 49): 'a photo of a person inspecting an orange',
                   (64, 49): 'a photo of a person peeling an orange',
                   (66, 49): 'a photo of a person picking an orange',
                   (91, 49): 'a photo of a person squeezing an orange',
                   (111, 49): 'a photo of a person washing an orange',
                   (57, 49): 'a photo of a person and an orange', (12, 69): 'a photo of a person cleaning an oven',
                   (36, 69): 'a photo of a person holding an oven',
                   (41, 69): 'a photo of a person inspecting an oven',
                   (58, 69): 'a photo of a person opening an oven',
                   (75, 69): 'a photo of a person repairing an oven',
                   (59, 69): 'a photo of a person operating an oven', (57, 69): 'a photo of a person and an oven',
                   (11, 12): 'a photo of a person checking a parking meter',
                   (63, 12): 'a photo of a person paying a parking meter',
                   (75, 12): 'a photo of a person repairing a parking meter',
                   (57, 12): 'a photo of a person and a parking meter',
                   (7, 53): 'a photo of a person buying a pizza', (8, 53): 'a photo of a person carrying a pizza',
                   (14, 53): 'a photo of a person cooking a pizza', (15, 53): 'a photo of a person cutting a pizza',
                   (23, 53): 'a photo of a person eating a pizza', (36, 53): 'a photo of a person holding a pizza',
                   (54, 53): 'a photo of a person making a pizza',
                   (67, 53): 'a photo of a person picking up a pizza',
                   (88, 53): 'a photo of a person sliding a pizza',
                   (89, 53): 'a photo of a person smelling a pizza', (57, 53): 'a photo of a person and a pizza',
                   (12, 72): 'a photo of a person cleaning a refrigerator',
                   (36, 72): 'a photo of a person holding a refrigerator',
                   (56, 72): 'a photo of a person moving a refrigerator',
                   (58, 72): 'a photo of a person opening a refrigerator',
                   (57, 72): 'a photo of a person and a refrigerator',
                   (36, 65): 'a photo of a person holding a remote',
                   (68, 65): 'a photo of a person pointing a remote',
                   (99, 65): 'a photo of a person swinging a remote', (57, 65): 'a photo of a person and a remote',
                   (8, 48): 'a photo of a person carrying a sandwich',
                   (14, 48): 'a photo of a person cooking a sandwich',
                   (15, 48): 'a photo of a person cutting a sandwich',
                   (23, 48): 'a photo of a person eating a sandwich',
                   (36, 48): 'a photo of a person holding a sandwich',
                   (54, 48): 'a photo of a person making a sandwich',
                   (57, 48): 'a photo of a person and a sandwich',
                   (16, 76): 'a photo of a person cutting with a scissors',
                   (36, 76): 'a photo of a person holding a scissors',
                   (58, 76): 'a photo of a person opening a scissors',
                   (57, 76): 'a photo of a person and a scissors', (12, 71): 'a photo of a person cleaning a sink',
                   (75, 71): 'a photo of a person repairing a sink',
                   (111, 71): 'a photo of a person washing a sink', (57, 71): 'a photo of a person and a sink',
                   (8, 36): 'a photo of a person carrying a skateboard',
                   (28, 36): 'a photo of a person flipping a skateboard',
                   (32, 36): 'a photo of a person grinding a skateboard',
                   (36, 36): 'a photo of a person holding a skateboard',
                   (43, 36): 'a photo of a person jumping a skateboard',
                   (67, 36): 'a photo of a person picking up a skateboard',
                   (76, 36): 'a photo of a person riding a skateboard',
                   (87, 36): 'a photo of a person sitting on a skateboard',
                   (93, 36): 'a photo of a person standing on a skateboard',
                   (57, 36): 'a photo of a person and a skateboard',
                   (0, 30): 'a photo of a person adjusting a skis', (8, 30): 'a photo of a person carrying a skis',
                   (36, 30): 'a photo of a person holding a skis',
                   (41, 30): 'a photo of a person inspecting a skis',
                   (43, 30): 'a photo of a person jumping a skis',
                   (67, 30): 'a photo of a person picking up a skis',
                   (75, 30): 'a photo of a person repairing a skis', (76, 30): 'a photo of a person riding a skis',
                   (93, 30): 'a photo of a person standing on a skis',
                   (114, 30): 'a photo of a person wearing a skis', (57, 30): 'a photo of a person and a skis',
                   (0, 31): 'a photo of a person adjusting a snowboard',
                   (8, 31): 'a photo of a person carrying a snowboard',
                   (32, 31): 'a photo of a person grinding a snowboard',
                   (36, 31): 'a photo of a person holding a snowboard',
                   (43, 31): 'a photo of a person jumping a snowboard',
                   (76, 31): 'a photo of a person riding a snowboard',
                   (93, 31): 'a photo of a person standing on a snowboard',
                   (114, 31): 'a photo of a person wearing a snowboard',
                   (57, 31): 'a photo of a person and a snowboard', (36, 44): 'a photo of a person holding a spoon',
                   (48, 44): 'a photo of a person licking a spoon',
                   (111, 44): 'a photo of a person washing a spoon',
                   (85, 44): 'a photo of a person sipping a spoon', (57, 44): 'a photo of a person and a spoon',
                   (2, 32): 'a photo of a person blocking a sports ball',
                   (8, 32): 'a photo of a person carrying a sports ball',
                   (9, 32): 'a photo of a person catching a sports ball',
                   (19, 32): 'a photo of a person dribbling a sports ball',
                   (35, 32): 'a photo of a person hitting a sports ball',
                   (36, 32): 'a photo of a person holding a sports ball',
                   (41, 32): 'a photo of a person inspecting a sports ball',
                   (44, 32): 'a photo of a person kicking a sports ball',
                   (67, 32): 'a photo of a person picking up a sports ball',
                   (81, 32): 'a photo of a person serving a sports ball',
                   (84, 32): 'a photo of a person signing a sports ball',
                   (90, 32): 'a photo of a person spinning a sports ball',
                   (104, 32): 'a photo of a person throwing a sports ball',
                   (57, 32): 'a photo of a person and a sports ball',
                   (36, 11): 'a photo of a person holding a stop sign',
                   (94, 11): 'a photo of a person standing under a stop sign',
                   (97, 11): 'a photo of a person stopping at a stop sign',
                   (57, 11): 'a photo of a person and a stop sign',
                   (8, 28): 'a photo of a person carrying a suitcase',
                   (18, 28): 'a photo of a person dragging a suitcase',
                   (36, 28): 'a photo of a person holding a suitcase',
                   (39, 28): 'a photo of a person hugging a suitcase',
                   (52, 28): 'a photo of a person loading a suitcase',
                   (58, 28): 'a photo of a person opening a suitcase',
                   (60, 28): 'a photo of a person packing a suitcase',
                   (67, 28): 'a photo of a person picking up a suitcase',
                   (116, 28): 'a photo of a person zipping a suitcase',
                   (57, 28): 'a photo of a person and a suitcase',
                   (8, 37): 'a photo of a person carrying a surfboard',
                   (18, 37): 'a photo of a person dragging a surfboard',
                   (36, 37): 'a photo of a person holding a surfboard',
                   (41, 37): 'a photo of a person inspecting a surfboard',
                   (43, 37): 'a photo of a person jumping a surfboard',
                   (49, 37): 'a photo of a person lying on a surfboard',
                   (52, 37): 'a photo of a person loading a surfboard',
                   (76, 37): 'a photo of a person riding a surfboard',
                   (93, 37): 'a photo of a person standing on a surfboard',
                   (87, 37): 'a photo of a person sitting on a surfboard',
                   (111, 37): 'a photo of a person washing a surfboard',
                   (57, 37): 'a photo of a person and a surfboard',
                   (8, 77): 'a photo of a person carrying a teddy bear',
                   (36, 77): 'a photo of a person holding a teddy bear',
                   (39, 77): 'a photo of a person hugging a teddy bear',
                   (45, 77): 'a photo of a person kissing a teddy bear',
                   (57, 77): 'a photo of a person and a teddy bear',
                   (8, 38): 'a photo of a person carrying a tennis racket',
                   (36, 38): 'a photo of a person holding a tennis racket',
                   (41, 38): 'a photo of a person inspecting a tennis racket',
                   (99, 38): 'a photo of a person swinging a tennis racket',
                   (57, 38): 'a photo of a person and a tennis racket',
                   (0, 27): 'a photo of a person adjusting a tie', (15, 27): 'a photo of a person cutting a tie',
                   (36, 27): 'a photo of a person holding a tie', (41, 27): 'a photo of a person inspecting a tie',
                   (70, 27): 'a photo of a person pulling a tie', (105, 27): 'a photo of a person tying a tie',
                   (114, 27): 'a photo of a person wearing a tie', (57, 27): 'a photo of a person and a tie',
                   (36, 70): 'a photo of a person holding a toaster',
                   (59, 70): 'a photo of a person operating a toaster',
                   (75, 70): 'a photo of a person repairing a toaster',
                   (57, 70): 'a photo of a person and a toaster', (12, 61): 'a photo of a person cleaning a toilet',
                   (29, 61): 'a photo of a person flushing a toilet',
                   (58, 61): 'a photo of a person opening a toilet',
                   (75, 61): 'a photo of a person repairing a toilet',
                   (87, 61): 'a photo of a person sitting on a toilet',
                   (93, 61): 'a photo of a person standing on a toilet',
                   (111, 61): 'a photo of a person washing a toilet', (57, 61): 'a photo of a person and a toilet',
                   (6, 79): 'a photo of a person brushing with a toothbrush',
                   (36, 79): 'a photo of a person holding a toothbrush',
                   (111, 79): 'a photo of a person washing a toothbrush',
                   (57, 79): 'a photo of a person and a toothbrush',
                   (42, 9): 'a photo of a person installing a traffic light',
                   (75, 9): 'a photo of a person repairing a traffic light',
                   (94, 9): 'a photo of a person standing under a traffic light',
                   (97, 9): 'a photo of a person stopping at a traffic light',
                   (57, 9): 'a photo of a person and a traffic light',
                   (17, 7): 'a photo of a person directing a truck', (21, 7): 'a photo of a person driving a truck',
                   (41, 7): 'a photo of a person inspecting a truck',
                   (52, 7): 'a photo of a person loading a truck', (75, 7): 'a photo of a person repairing a truck',
                   (76, 7): 'a photo of a person riding a truck', (87, 7): 'a photo of a person sitting on a truck',
                   (111, 7): 'a photo of a person washing a truck', (57, 7): 'a photo of a person and a truck',
                   (8, 25): 'a photo of a person carrying a umbrella',
                   (36, 25): 'a photo of a person holding a umbrella',
                   (53, 25): 'a photo of a person losing a umbrella',
                   (58, 25): 'a photo of a person opening a umbrella',
                   (75, 25): 'a photo of a person repairing a umbrella',
                   (82, 25): 'a photo of a person setting a umbrella',
                   (94, 25): 'a photo of a person standing under a umbrella',
                   (57, 25): 'a photo of a person and a umbrella', (36, 75): 'a photo of a person holding a vase',
                   (54, 75): 'a photo of a person making a vase', (61, 75): 'a photo of a person painting a vase',
                   (57, 75): 'a photo of a person and a vase', (27, 40): 'a photo of a person filling a wine glass',
                   (36, 40): 'a photo of a person holding a wine glass',
                   (85, 40): 'a photo of a person sipping a wine glass',
                   (106, 40): 'a photo of a person toasting a wine glass',
                   (48, 40): 'a photo of a person licking a wine glass',
                   (111, 40): 'a photo of a person washing a wine glass',
                   (57, 40): 'a photo of a person and a wine glass',
                   (26, 22): 'a photo of a person feeding a zebra', (36, 22): 'a photo of a person holding a zebra',
                   (65, 22): 'a photo of a person petting a zebra',
                   (112, 22): 'a photo of a person watching a zebra', (57, 22): 'a photo of a person and a zebra'}

def sift(li, low, higt):
    tmp = li[low]
    i = low
    j = 2 * i + 1
    while j <= higt:  # 情况2：i已经是最后一层
        if j + 1 <= higt and li[j + 1] < li[j]:  # 右孩子存在并且小于左孩子
            j += 1
        if tmp > li[j]:
            li[i] = li[j]
            i = j
            j = 2 * i + 1
        else:
            break  # 情况1：j位置比tmp小
    li[i] = tmp

def top_k(li, k):
    heap = li[0:k]
    # 建堆
    for i in range(k // 2 - 1, -1, -1):
        sift(heap, i, k - 1)
    for i in range(k, len(li)):
        if li[i] > heap[0]:
            heap[0] = li[i]
            sift(heap, 0, k - 1)
    # 挨个输出
    for i in range(k - 1, -1, -1):
        heap[0], heap[i] = heap[i], heap[0]
        sift(heap, 0, i - 1)

    return heap

class HICOEvaluator():
    def __init__(self, preds, gts, rare_triplets, non_rare_triplets, correct_mat, args):
        self.overlap_iou = 0.5
        self.max_hois = 100

        self.zero_shot_type = args.zero_shot_type

        self.use_nms_filter = args.use_nms_filter
        self.thres_nms = args.thres_nms
        self.nms_alpha = args.nms_alpha
        self.nms_beta = args.nms_beta

        self.use_score_thres = False
        self.thres_score = 1e-5

        self.use_soft_nms = False
        self.soft_nms_sigma = 0.5
        self.soft_nms_thres_score = 1e-11

        self.rare_triplets = rare_triplets
        self.non_rare_triplets = non_rare_triplets

        self.fp = defaultdict(list)
        self.tp = defaultdict(list)
        self.score = defaultdict(list)
        self.sum_gts = defaultdict(lambda: 0)
        self.gt_triplets = []

        self.preds = []
        self.hico_triplet_labels = list(hico_text_label.keys())
        print(self.hico_triplet_labels)
        self.hoi_obj_list = []
        for hoi_pair in self.hico_triplet_labels:
            self.hoi_obj_list.append(hoi_pair[1])

        for index, img_preds in enumerate(preds):
            img_preds = {k: v.to('cpu').numpy() for k, v in img_preds.items()}
            bboxes = [{'bbox': list(bbox)} for bbox in img_preds['boxes']]
            hoi_scores = img_preds['verb_scores']
            hoi_labels = np.tile(np.arange(hoi_scores.shape[1]), (hoi_scores.shape[0], 1))
            subject_ids = np.tile(img_preds['sub_ids'], (hoi_scores.shape[1], 1)).T
            object_ids = np.tile(img_preds['obj_ids'], (hoi_scores.shape[1], 1)).T

            hoi_labels = np.tile(np.arange(hoi_scores.shape[1]), (hoi_scores.shape[0], 1))
            subject_ids = np.tile(img_preds['sub_ids'], (hoi_scores.shape[1], 1)).T
            object_ids = np.tile(img_preds['obj_ids'], (hoi_scores.shape[1], 1)).T

            hoi_scores = hoi_scores.ravel()
            hoi_labels = hoi_labels.ravel()
            subject_ids = subject_ids.ravel()
            object_ids = object_ids.ravel()

            topk_hoi_scores = top_k(list(hoi_scores), self.max_hois)
            topk_indexes = np.array([np.where(hoi_scores == score)[0][0] for score in topk_hoi_scores])

            if len(subject_ids) > 0:
                hois = [{'subject_id': subject_id, 'object_id': object_id, 'category_id': category_id, 'score': score}
                        for
                        subject_id, object_id, category_id, score in
                        zip(subject_ids[topk_indexes], object_ids[topk_indexes], hoi_labels[topk_indexes], topk_hoi_scores)]
                hois = hois[:self.max_hois]
            else:
                hois = []

            filename = gts[index]['filename']
            self.preds.append({
                'filename': filename,
                'predictions': bboxes,
                'hoi_prediction': hois
            })

        if self.use_nms_filter:
            print('eval use_nms_filter ...')
            self.preds = self.triplet_nms_filter(self.preds)


        self.gts = []

        for i, img_gts in enumerate(gts):
            filename = img_gts['filename']
            img_gts = {k: v.to('cpu').numpy() for k, v in img_gts.items() if k != 'id' and k != 'filename'}
            bbox_anns = [{'bbox': list(bbox), 'category_id': label} for bbox, label in
                         zip(img_gts['boxes'], img_gts['labels'])]
            hoi_anns = [{'subject_id': hoi[0], 'object_id': hoi[1],
                         'category_id': self.hico_triplet_labels.index((hoi[2], bbox_anns[hoi[1]]['category_id']))}
                        for hoi in img_gts['hois']]
            self.gts.append({
                'filename': filename,
                'annotations': bbox_anns,
                'hoi_annotation': hoi_anns
            })
            for hoi in self.gts[-1]['hoi_annotation']:
                triplet = hoi['category_id']

                if triplet not in self.gt_triplets:
                    self.gt_triplets.append(triplet)

                self.sum_gts[triplet] += 1

        with open(args.json_file, 'w') as f:
            f.write(json.dumps(str({'preds': self.preds, 'gts': self.gts})))

    def evaluate(self):
        for img_preds, img_gts in zip(self.preds, self.gts):
            pred_bboxes = img_preds['predictions']
            if len(pred_bboxes) == 0: continue

            gt_bboxes = img_gts['annotations']
            pred_hois = img_preds['hoi_prediction']
            gt_hois = img_gts['hoi_annotation']
            if len(gt_bboxes) != 0:
                bbox_pairs, bbox_overlaps = self.compute_iou_mat(gt_bboxes, pred_bboxes)
                self.compute_fptp(pred_hois, gt_hois, bbox_pairs, bbox_overlaps)
            else:
                for pred_hoi in pred_hois:
                    triplet = pred_hoi['category_id']
                    if triplet not in self.gt_triplets:
                        continue
                    self.tp[triplet].append(0)
                    self.fp[triplet].append(1)
                    self.score[triplet].append(pred_hoi['score'])
        map = self.compute_map()
        return map

    def compute_map(self):
        ap = defaultdict(lambda: 0)
        rare_ap = defaultdict(lambda: 0)
        non_rare_ap = defaultdict(lambda: 0)
        max_recall = defaultdict(lambda: 0)
        count = 0
        for triplet in self.gt_triplets:
            print(triplet)
            sum_gts = self.sum_gts[triplet]
            if sum_gts == 0:
                continue

            tp = np.array((self.tp[triplet]))
            fp = np.array((self.fp[triplet]))
            if len(tp) == 0:
                ap[triplet] = 0
                max_recall[triplet] = 0
                if triplet in self.rare_triplets:
                    rare_ap[triplet] = 0
                elif triplet in self.non_rare_triplets:
                    non_rare_ap[triplet] = 0
                else:
                    print('Warning: triplet {} is neither in rare triplets nor in non-rare triplets'.format(triplet))
                    count += 1
                continue

            score = np.array(self.score[triplet])
            sort_inds = np.argsort(-score)
            fp = fp[sort_inds]
            tp = tp[sort_inds]
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / sum_gts
            prec = tp / (fp + tp)
            # ap[triplet] = self.cal_prec(rec, prec)
            ap[triplet] = self.voc_ap(rec, prec)
            max_recall[triplet] = np.amax(rec)
            orignal_triplet = self.hico_triplet_labels[triplet]
            orignal_triplet = (0, orignal_triplet[1], orignal_triplet[0])
            if orignal_triplet in self.rare_triplets:
                rare_ap[triplet] = ap[triplet]
            elif orignal_triplet in self.non_rare_triplets:
                non_rare_ap[triplet] = ap[triplet]
            else:
                print('Warning: triplet {} is neither in rare triplets nor in non-rare triplets'.format(triplet))
                count += 1
        m_ap = np.mean(list(ap.values()))
        m_ap_rare = np.mean(list(rare_ap.values()))
        m_ap_non_rare = np.mean(list(non_rare_ap.values()))
        m_max_recall = np.mean(list(max_recall.values()))
        print(count)

        print('--------------------')
        if self.zero_shot_type == "default":
            print('mAP full: {} mAP rare: {}  mAP non-rare: {}  mean max recall: {}'.format(m_ap, m_ap_rare, m_ap_non_rare,
                                                                                    m_max_recall))
            return_dict = {'mAP': m_ap, 'mAP rare': m_ap_rare, 'mAP non-rare': m_ap_non_rare, 'mean max recall': m_max_recall}

        elif self.zero_shot_type == "unseen_object":
            print('mAP full: {} mAP unseen-obj: {}  mAP seen-obj: {}  mean max recall: {}'.format(m_ap, m_ap_rare, m_ap_non_rare,
                                                                                    m_max_recall))
            return_dict = {'mAP': m_ap, 'mAP unseen-obj': m_ap_rare, 'mAP seen-obj': m_ap_non_rare, 'mean max recall': m_max_recall}

        else:
            print('mAP full: {} mAP unseen: {}  mAP seen: {}  mean max recall: {}'.format(m_ap, m_ap_rare, m_ap_non_rare,
                                                                                    m_max_recall))
            return_dict = {'mAP': m_ap, 'mAP unseen': m_ap_rare, 'mAP seen': m_ap_non_rare, 'mean max recall': m_max_recall}

        print('--------------------')

        return return_dict

    def cal_prec(self, rec, prec, t=0.8):
        if np.sum(rec >= t) == 0:
            p = 0
        else:
            p = np.max(prec[rec >= t])
        return p

    def voc_ap(self, rec, prec):
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
        return ap

    def compute_fptp(self, pred_hois, gt_hois, match_pairs, bbox_overlaps):
        pos_pred_ids = match_pairs.keys()
        vis_tag = np.zeros(len(gt_hois))
        pred_hois.sort(key=lambda k: (k.get('score', 0)), reverse=True)
        if len(pred_hois) != 0:
            for pred_hoi in pred_hois:
                is_match = 0
                if len(match_pairs) != 0 and pred_hoi['subject_id'] in pos_pred_ids and pred_hoi[
                    'object_id'] in pos_pred_ids:
                    pred_sub_ids = match_pairs[pred_hoi['subject_id']]
                    pred_obj_ids = match_pairs[pred_hoi['object_id']]
                    pred_sub_overlaps = bbox_overlaps[pred_hoi['subject_id']]
                    pred_obj_overlaps = bbox_overlaps[pred_hoi['object_id']]
                    pred_category_id = pred_hoi['category_id']
                    max_overlap = 0
                    max_gt_hoi = 0
                    for gt_hoi in gt_hois:
                        if gt_hoi['subject_id'] in pred_sub_ids and gt_hoi['object_id'] in pred_obj_ids \
                                and pred_category_id == gt_hoi['category_id']:
                            is_match = 1
                            min_overlap_gt = min(pred_sub_overlaps[pred_sub_ids.index(gt_hoi['subject_id'])],
                                                 pred_obj_overlaps[pred_obj_ids.index(gt_hoi['object_id'])])
                            if min_overlap_gt > max_overlap:
                                max_overlap = min_overlap_gt
                                max_gt_hoi = gt_hoi
                triplet = pred_hoi['category_id']
                if triplet not in self.gt_triplets:
                    continue
                if is_match == 1 and vis_tag[gt_hois.index(max_gt_hoi)] == 0:
                    self.fp[triplet].append(0)
                    self.tp[triplet].append(1)
                    vis_tag[gt_hois.index(max_gt_hoi)] = 1
                else:
                    self.fp[triplet].append(1)
                    self.tp[triplet].append(0)
                self.score[triplet].append(pred_hoi['score'])

    def compute_iou_mat(self, bbox_list1, bbox_list2):
        iou_mat = np.zeros((len(bbox_list1), len(bbox_list2)))
        if len(bbox_list1) == 0 or len(bbox_list2) == 0:
            return {}
        for i, bbox1 in enumerate(bbox_list1):
            for j, bbox2 in enumerate(bbox_list2):
                iou_i = self.compute_IOU(bbox1, bbox2)
                iou_mat[i, j] = iou_i

        iou_mat_ov = iou_mat.copy()
        iou_mat[iou_mat >= self.overlap_iou] = 1
        iou_mat[iou_mat < self.overlap_iou] = 0

        match_pairs = np.nonzero(iou_mat)
        match_pairs_dict = {}
        match_pair_overlaps = {}
        if iou_mat.max() > 0:
            for i, pred_id in enumerate(match_pairs[1]):
                if pred_id not in match_pairs_dict.keys():
                    match_pairs_dict[pred_id] = []
                    match_pair_overlaps[pred_id] = []
                match_pairs_dict[pred_id].append(match_pairs[0][i])
                match_pair_overlaps[pred_id].append(iou_mat_ov[match_pairs[0][i], pred_id])
        return match_pairs_dict, match_pair_overlaps

    def compute_IOU(self, bbox1, bbox2):
        rec1 = bbox1['bbox']
        rec2 = bbox2['bbox']
        # computing area of each rectangles
        S_rec1 = (rec1[2] - rec1[0] + 1) * (rec1[3] - rec1[1] + 1)
        S_rec2 = (rec2[2] - rec2[0] + 1) * (rec2[3] - rec2[1] + 1)

        # computing the sum_area
        sum_area = S_rec1 + S_rec2

        # find the each edge of intersect rectangle
        left_line = max(rec1[1], rec2[1])
        right_line = min(rec1[3], rec2[3])
        top_line = max(rec1[0], rec2[0])
        bottom_line = min(rec1[2], rec2[2])
        # judge if there is an intersect
        if left_line >= right_line or top_line >= bottom_line:
            return 0
        else:
            intersect = (right_line - left_line + 1) * (bottom_line - top_line + 1)
            return intersect / (sum_area - intersect)

    def triplet_nms_filter(self, preds):
        preds_filtered = []
        for img_preds in preds:
            pred_bboxes = img_preds['predictions']
            pred_hois = img_preds['hoi_prediction']
            all_triplets = {}
            for index, pred_hoi in enumerate(pred_hois):
                triplet = pred_hoi['category_id']

                if triplet not in all_triplets:
                    all_triplets[triplet] = {'subs': [], 'objs': [], 'scores': [], 'indexes': []}
                all_triplets[triplet]['subs'].append(pred_bboxes[pred_hoi['subject_id']]['bbox'])
                all_triplets[triplet]['objs'].append(pred_bboxes[pred_hoi['object_id']]['bbox'])
                all_triplets[triplet]['scores'].append(pred_hoi['score'])
                all_triplets[triplet]['indexes'].append(index)

            all_keep_inds = []
            for triplet, values in all_triplets.items():
                subs, objs, scores = values['subs'], values['objs'], values['scores']
                if self.use_soft_nms:
                    keep_inds = self.pairwise_soft_nms(np.array(subs), np.array(objs), np.array(scores))
                else:
                    keep_inds = self.pairwise_nms(np.array(subs), np.array(objs), np.array(scores))

                if self.use_score_thres:
                    sorted_scores = np.array(scores)[keep_inds]
                    keep_inds = np.array(keep_inds)[sorted_scores > self.thres_score]

                keep_inds = list(np.array(values['indexes'])[keep_inds])
                all_keep_inds.extend(keep_inds)

            preds_filtered.append({
                'filename': img_preds['filename'],
                'predictions': pred_bboxes,
                'hoi_prediction': list(np.array(img_preds['hoi_prediction'])[all_keep_inds])
            })

        return preds_filtered

    def pairwise_nms(self, subs, objs, scores):
        sx1, sy1, sx2, sy2 = subs[:, 0], subs[:, 1], subs[:, 2], subs[:, 3]
        ox1, oy1, ox2, oy2 = objs[:, 0], objs[:, 1], objs[:, 2], objs[:, 3]

        sub_areas = (sx2 - sx1 + 1) * (sy2 - sy1 + 1)
        obj_areas = (ox2 - ox1 + 1) * (oy2 - oy1 + 1)

        order = scores.argsort()[::-1]

        keep_inds = []
        while order.size > 0:
            i = order[0]
            keep_inds.append(i)

            sxx1 = np.maximum(sx1[i], sx1[order[1:]])
            syy1 = np.maximum(sy1[i], sy1[order[1:]])
            sxx2 = np.minimum(sx2[i], sx2[order[1:]])
            syy2 = np.minimum(sy2[i], sy2[order[1:]])

            sw = np.maximum(0.0, sxx2 - sxx1 + 1)
            sh = np.maximum(0.0, syy2 - syy1 + 1)
            sub_inter = sw * sh
            sub_union = sub_areas[i] + sub_areas[order[1:]] - sub_inter

            oxx1 = np.maximum(ox1[i], ox1[order[1:]])
            oyy1 = np.maximum(oy1[i], oy1[order[1:]])
            oxx2 = np.minimum(ox2[i], ox2[order[1:]])
            oyy2 = np.minimum(oy2[i], oy2[order[1:]])

            ow = np.maximum(0.0, oxx2 - oxx1 + 1)
            oh = np.maximum(0.0, oyy2 - oyy1 + 1)
            obj_inter = ow * oh
            obj_union = obj_areas[i] + obj_areas[order[1:]] - obj_inter

            ovr = np.power(sub_inter / sub_union, self.nms_alpha) * np.power(obj_inter / obj_union, self.nms_beta)
            inds = np.where(ovr <= self.thres_nms)[0]

            order = order[inds + 1]
        return keep_inds

    def pairwise_soft_nms(self, subs, objs, scores):
        assert subs.shape[0] == objs.shape[0]
        N = subs.shape[0]

        sx1, sy1, sx2, sy2 = subs[:, 0], subs[:, 1], subs[:, 2], subs[:, 3]
        ox1, oy1, ox2, oy2 = objs[:, 0], objs[:, 1], objs[:, 2], objs[:, 3]

        sub_areas = (sx2 - sx1 + 1) * (sy2 - sy1 + 1)
        obj_areas = (ox2 - ox1 + 1) * (oy2 - oy1 + 1)

        for i in range(N):
            tscore = scores[i]
            pos = i + 1
            if i != N - 1:
                maxpos = np.argmax(scores[pos:])
                maxscore = scores[pos:][maxpos]

                if tscore < maxscore:
                    subs[i], subs[maxpos.item() + i + 1] = subs[maxpos.item() + i + 1].copy(), subs[i].copy()
                    sub_areas[i], sub_areas[maxpos + i + 1] = sub_areas[maxpos + i + 1].copy(), sub_areas[i].copy()

                    objs[i], objs[maxpos.item() + i + 1] = objs[maxpos.item() + i + 1].copy(), objs[i].copy()
                    obj_areas[i], obj_areas[maxpos + i + 1] = obj_areas[maxpos + i + 1].copy(), obj_areas[i].copy()

                    scores[i], scores[maxpos.item() + i + 1] = scores[maxpos.item() + i + 1].copy(), scores[i].copy()

            # IoU calculate
            sxx1 = np.maximum(subs[i, 0], subs[pos:, 0])
            syy1 = np.maximum(subs[i, 1], subs[pos:, 1])
            sxx2 = np.minimum(subs[i, 2], subs[pos:, 2])
            syy2 = np.minimum(subs[i, 3], subs[pos:, 3])

            sw = np.maximum(0.0, sxx2 - sxx1 + 1)
            sh = np.maximum(0.0, syy2 - syy1 + 1)
            sub_inter = sw * sh
            sub_union = sub_areas[i] + sub_areas[pos:] - sub_inter
            sub_ovr = sub_inter / sub_union

            oxx1 = np.maximum(objs[i, 0], objs[pos:, 0])
            oyy1 = np.maximum(objs[i, 1], objs[pos:, 1])
            oxx2 = np.minimum(objs[i, 2], objs[pos:, 2])
            oyy2 = np.minimum(objs[i, 3], objs[pos:, 3])

            ow = np.maximum(0.0, oxx2 - oxx1 + 1)
            oh = np.maximum(0.0, oyy2 - oyy1 + 1)
            obj_inter = ow * oh
            obj_union = obj_areas[i] + obj_areas[pos:] - obj_inter
            obj_ovr = obj_inter / obj_union

            # Gaussian decay
            ## mode 1
            # weight = np.exp(-(sub_ovr * obj_ovr) / self.soft_nms_sigma)

            ## mode 2
            weight = np.exp(-sub_ovr / self.soft_nms_sigma) * np.exp(-obj_ovr / self.soft_nms_sigma)

            scores[pos:] = weight * scores[pos:]

        # select the boxes and keep the corresponding indexes
        keep_inds = np.where(scores > self.soft_nms_thres_score)[0]

        return keep_inds

    def clip_preds_boxes(self, preds):
        preds_filtered = []
        for img_preds in preds:
            filename = img_preds['filename']

            input_file = os.path.join('data/hico_20160224_det/images/test2015/', filename)
            img = cv2.imread(input_file)
            h, w, c = img.shape

            pred_bboxes = img_preds['predictions']
            for pred_bbox in pred_bboxes:
                pred_bbox['bbox'] = self.bbox_clip(pred_bbox['bbox'], (h, w))

            preds_filtered.append(img_preds)

        return preds_filtered

    def bbox_clip(self, box, size):
        x1, y1, x2, y2 = box
        h, w = size
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(x2, w)
        y2 = min(y2, h)
        return [x1, y1, x2, y2]