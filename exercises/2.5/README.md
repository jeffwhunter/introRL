<p align="center">
    <img src="./2.5.png">
    <br/>
    <em>
        Plots of the reward and chance of optimal action averaged over 2000 agents
        learning to play 10 slot machines with payouts normally distributed around zero.
        The value e controls how much each agent explores: .1 implies the agent explores
        10% of the time and otherwise takes the best expected action. Slot machines in
        columns titled 'walk' will slowly change their average payout each turn. Agents
        in columns titled '1/N step' will value actions according to the average return
        of those actions; in columns titled '0.1 step' they will use a weighted average
        return instead, preferring more recent values.
    </em>
</p>