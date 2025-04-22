from direct.fsm.FSM import FSM

class RewardOrPuff(FSM):
    """
    FSM to manage the reward or puff state.
    """
    def __init__(self, base: ShowBase, config: Dict[str, Any]) -> None:
        """
        Initialize the FSM with the base and configuration.

        Parameters:
            base (ShowBase): The Panda3D base instance.
            config (dict): Configuration parameters.
        """
        FSM.__init__(self, "RewardOrPuff")
        self.base = base
        self.config = config
        self.accept('puff-event', self.request, ['Puff'])
        self.accept('reward-event', self.request, ['Reward'])
        self.accept('neutral-event', self.request, ['Neutral'])
        #zone_time = elapsed_time

    def enterPuff(self):
        """
        Enter the Puff state.
        """
        print("Entering Puff state")
        taskMgr.doMethodLater(1.0, self._transitionToNeutral, 'return-to-neutral')

    def exitPuff(self):
        """
        Exit the Puff state.
        """
        print("Exiting Puff state")
        
    def enterReward(self):
        print("Entering Reward state: give reward (e.g. juice drop)")
        taskMgr.doMethodLater(1.0, self._transitionToNeutral, 'return-to-neutral')

    def exitReward(self):
        print("Exiting Reward state.")

    def enterNeutral(self):
        print("Entering Neutral state: waiting...")

    def exitNeutral(self):
        print("Exiting Neutral state.")

    def _transitionToNeutral(self, task):
        self.request('Neutral')
        return Task.done

        

if virtual_distance < 5 and zone_time > elapsed_time + 2.0 and selected_texture == self.special_wall:
    fsm.request('Reward')
elif virtual_distance > 5 and zone_time > elapsed_time + 2.0 and selected_texture == self.alternative_wall_texture_2:
    fsm.request('Puff')
else:
    fsm.request('Neutral')