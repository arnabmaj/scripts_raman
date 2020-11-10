import torch
from mlcalcdriver.calculators.utils import torch_derivative


def simple_fn(args):
    def loss(batch, result):
        diff = (batch[args.property] - result[args.property]) ** 2
        err_sq = torch.mean(diff)
        return err_sq

    return loss


def tilted_down(args):
    def loss(batch, result):
        diff = batch[args.property] - result[args.property]
        idx = torch.where(diff >= 0)
        diff[idx] = diff[idx] * 4.0
        err = torch.mean(diff ** 2)
        return err

    return loss


def tilted_up(args):
    def loss(batch, result):
        diff = batch[args.property] - result[args.property]
        idx = torch.where(diff <= 0)
        diff[idx] = diff[idx] * 4.0
        err = torch.mean(diff ** 2)
        return err

    return loss


def tradeoff_loss_fn(rho, property_names):
    def loss_fn(batch, result):
        loss = 0
        for prop in rho.keys():
            diff = (batch[property_names[prop]] - result[property_names[prop]]) ** 2
            err_sq = rho[prop] * torch.mean(diff)
            loss += err_sq
        return loss

    return loss_fn


def smooth_loss(args, rho, property_names):
    def loss_fn_no_derivative(batch, result):
        err = torch.mean((batch[args.property] - result[args.property]) ** 2)
        deriv1 = torch.unsqueeze(
            torch_derivative(results[args.property], batch["_positions"]), 0,
        )
        random_vector = torch.rand(batch["_positions"].shape) * 0.02 - 0.01
        deriv2 = torch.unsqueeze(
            torch_derivative(
                results[args.property], batch["_positions"] + random_vector
            ),
            0,
        )
        loss = (
            err
            + args.smooth_lambda
            / args.batch_size
            * (torch.linalg.norm(deriv1 - deriv2)) ** 2
        )
        return loss

    def loss_fn_derivative(batch, result):
        loss = 0
        for prop in rho.keys():
            diff = (batch[property_names[prop]] - result[property_names[prop]]) ** 2
            loss += rho[prop] * torch.mean(diff)
        deriv1 = batch

        # À REVOIR
