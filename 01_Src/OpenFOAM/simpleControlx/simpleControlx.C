/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2011-2012 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "simpleControlx.H"
#include "Time.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(simpleControlx, 0);
}


// * * * * * * * * * * * * Protected Member Functions  * * * * * * * * * * * //

void Foam::simpleControlx::read()
{
    solutionControlx::read(true);
}


bool Foam::simpleControlx::criteriaSatisfied()
{
    if (residualControl_.empty())
    {
        return false;
    }

    bool achieved = true;
    bool checked = false;    // safety that some checks were indeed performed

    // get iteration and lastResiduals index
    label iterNum = static_cast<label>(mesh_.time().value() / mesh_.time().deltaTValue());
    label resIdx = (iterNum-1) % nMonitorIter_;
    
    const dictionary& solverDict = mesh_.solverPerformanceDict();
    forAllConstIter(dictionary, solverDict, iter)
    {
        const word& variableName = iter().keyword();
        const label fieldI = applyToField(variableName);
        if (fieldI != -1)
        {
            const List<solverPerformance> sp(iter().stream());
            const scalar residual = sp.first().initialResidual();

            // write residual
            residualControl_[fieldI].lastResiduals[resIdx] = residual;

            // check convergence
            if (iterNum > nMonitorIter_)
            {
                // get min/max/mean values
                scalar resMin = 9999.0;
                scalar resMax = 0.0;
                scalar resMean = 0.0;

                forAll(residualControl_[fieldI].lastResiduals, resI)
                {
                    resMin = min(resMin, residualControl_[fieldI].lastResiduals[resI]);
                    resMax = max(resMax, residualControl_[fieldI].lastResiduals[resI]);
                    resMean += residualControl_[fieldI].lastResiduals[resI];
                }

                resMean /= nMonitorIter_;

                bool meanCheck = resMean < residualControl_[fieldI].absTol;
                bool minCheck = resMin >= (resMean / convFac_);
                bool maxCheck = resMax <= (resMean * (2.0 - (1.0 / convFac_)));

                checked = true;

                achieved = achieved && meanCheck && minCheck && maxCheck;
            }
            
            if (debug)
            {
                Info<< algorithmName_ << " solution statistics:" << endl;

                Info<< "    " << variableName << ": tolerance = " << residual
                    << " (" << residualControl_[fieldI].absTol << ")"
                    << endl;
            }
        }
    }

    return checked && achieved;
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::simpleControlx::simpleControlx(fvMesh& mesh)
:
    solutionControlx(mesh, "SIMPLE"),
    initialised_(false)
{
    read();

    Info<< nl;

    if (residualControl_.empty())
    {
        Info<< algorithmName_ << ": no convergence criteria found. "
            << "Calculations will run for " << mesh_.time().endTime().value()
            << " steps." << nl << endl;
    }
    else
    {
        Info<< algorithmName_ << ": convergence criteria" << nl;
        forAll(residualControl_, i)
        {
            Info<< "    field " << residualControl_[i].name << token::TAB
                << " tolerance " << residualControl_[i].absTol
                << nl;
        }
        Info<< endl;
    }
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::simpleControlx::~simpleControlx()
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

bool Foam::simpleControlx::loop()
{
    read();

    Time& time = const_cast<Time&>(mesh_.time());

    if (initialised_)
    {
        if (criteriaSatisfied())
        {
            Info<< nl << algorithmName_ << " solution converged in "
                << time.timeName() << " iterations" << nl << endl;

            // Set to finalise calculation
            time.writeAndEnd();
        }
        else
        {
            storePrevIterFields();
        }
    }
    else
    {
        initialised_ = true;
        storePrevIterFields();
    }


    return time.loop();
}


// ************************************************************************* //
