import unittest

import torch

from models.dmg_poe_calsurv import CalibrationLoss


class CalibrationLossTest(unittest.TestCase):
    def test_predictions_after_censoring_do_not_affect_loss(self):
        loss_fn = CalibrationLoss(num_time_bins=2)
        events = torch.tensor([0.0, 0.0, 1.0])
        times = torch.tensor([10.0, 4.0, 4.0])

        survival_a = torch.tensor(
            [[1.0, 0.2], [1.0, 0.0], [1.0, 0.0]], requires_grad=True
        )
        survival_b = torch.tensor(
            [[1.0, 0.9], [1.0, 0.8], [1.0, 0.0]], requires_grad=True
        )

        loss_a = loss_fn.compute_brier_calibration(
            survival_a, events, times, max_time=10.0
        )
        loss_b = loss_fn.compute_brier_calibration(
            survival_b, events, times, max_time=10.0
        )

        self.assertTrue(torch.allclose(loss_a, loss_b))

    def test_observed_event_status_affects_loss(self):
        loss_fn = CalibrationLoss(num_time_bins=2)
        events = torch.tensor([0.0, 0.0, 1.0])
        times = torch.tensor([10.0, 4.0, 4.0])

        correct_event_prediction = torch.tensor(
            [[1.0, 0.2], [1.0, 0.8], [1.0, 0.0]]
        )
        incorrect_event_prediction = torch.tensor(
            [[1.0, 0.2], [1.0, 0.8], [1.0, 1.0]]
        )

        correct_loss = loss_fn.compute_brier_calibration(
            correct_event_prediction, events, times, max_time=10.0
        )
        incorrect_loss = loss_fn.compute_brier_calibration(
            incorrect_event_prediction, events, times, max_time=10.0
        )

        self.assertLess(correct_loss.item(), incorrect_loss.item())


if __name__ == '__main__':
    unittest.main()
