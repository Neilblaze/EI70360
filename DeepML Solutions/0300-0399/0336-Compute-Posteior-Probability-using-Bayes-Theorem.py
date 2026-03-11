def bayes_theorem(priors: list[float], likelihoods: list[float]) -> list[float]:
	"""
	Calculate posterior probabilities using Bayes' Theorem.
	
	Args:
		priors: Prior probabilities P(H_i) for each hypothesis
		likelihoods: Likelihoods P(E|H_i) for each hypothesis
		
	Returns:
		Posterior probabilities P(H_i|E) for each hypothesis
	"""
	numerators = [p * l for p, l in zip(priors, likelihoods)]
	evidence = sum(numerators)
	if evidence == 0:
		return None

	posteriors = [round(n / evidence, 3) for n in numerators]
	return posteriors
