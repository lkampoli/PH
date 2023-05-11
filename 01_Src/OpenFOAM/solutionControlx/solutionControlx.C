/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2011 OpenFOAM Foundation
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

#include "solutionControlx.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(solutionControlx, 0);
}


// * * * * * * * * * * * * Protected Member Functions  * * * * * * * * * * * //

void Foam::solutionControlx::read(const bool absTolOnly)
{
    const dictionary& solnDict = this->dict();

    // Read solution controls
    nNonOrthCorr_ =
        solnDict.lookupOrDefault<label>("nNonOrthogonalCorrectors", 0);
    momentumPredictor_ = solnDict.lookupOrDefault("momentumPredictor", true);
    transonic_ = solnDict.lookupOrDefault("transonic", false);

    // Read convergence controls
    nMonitorIter_ = solnDict.lookupOrDefault<label>("nMonitorIter", 1);
    convFac_ = solnDict.lookupOrDefault<scalar>("convFac", 1.0);

    List<scalar> initList(nMonitorIter_, 9999.0);

    // Read residual information
    const dictionary residualDict(solnDict.subOrEmptyDict("residualControl"));

    DynamicList<fieldData> data(residualControl_);

    forAllConstIter(dictionary, residualDict, iter)
    {
        const word& fName = iter().keyword();
        const label fieldI = applyToField(fName, false);
        if (fieldI == -1)
        {
            fieldData fd;
            fd.name = fName.c_str();

            if (absTolOnly)
            {
                fd.absTol = readScalar(residualDict.lookup(fName));
                fd.relTol = -1;
                fd.initialResidual = -1;
                fd.lastResiduals = initList;
            }
            else
            {
                if (iter().isDict())
                {
                    const dictionary& fieldDict(iter().dict());
                    fd.absTol = readScalar(fieldDict.lookup("tolerance"));
                    fd.relTol = readScalar(fieldDict.lookup("relTol"));
                    fd.initialResidual = 0.0;
                }
                else
                {
                    FatalErrorIn("bool Foam::solutionControlx::read()")
                        << "Residual data for " << iter().keyword()
                        << " must be specified as a dictionary"
                        << exit(FatalError);
                }
            }

            data.append(fd);
        }
        else
        {
            fieldData& fd = data[fieldI];
            if (absTolOnly)
            {
                fd.absTol = readScalar(residualDict.lookup(fName));
            }
            else
            {
                if (iter().isDict())
                {
                    const dictionary& fieldDict(iter().dict());
                    fd.absTol = readScalar(fieldDict.lookup("tolerance"));
                    fd.relTol = readScalar(fieldDict.lookup("relTol"));
                }
                else
                {
                    FatalErrorIn("bool Foam::solutionControlx::read()")
                        << "Residual data for " << iter().keyword()
                        << " must be specified as a dictionary"
                        << exit(FatalError);
                }
            }
        }
    }

    residualControl_.transfer(data);

    if (debug)
    {
        forAll(residualControl_, i)
        {
            const fieldData& fd = residualControl_[i];
            Info<< "residualControl[" << i << "]:" << nl
                << "    name     : " << fd.name << nl
                << "    absTol   : " << fd.absTol << nl
                << "    relTol   : " << fd.relTol << nl
                << "    iniResid : " << fd.initialResidual << endl;
        }
    }
}


Foam::label Foam::solutionControlx::applyToField
(
    const word& fieldName,
    const bool useRegEx
) const
{
    forAll(residualControl_, i)
    {
        if (useRegEx && residualControl_[i].name.match(fieldName))
        {
            return i;
        }
        else if (residualControl_[i].name == fieldName)
        {
            return i;
        }
    }

    return -1;
}


void Foam::solutionControlx::storePrevIterFields() const
{
//    storePrevIter<label>();
    storePrevIter<scalar>();
    storePrevIter<vector>();
    storePrevIter<sphericalTensor>();
    storePrevIter<symmTensor>();
    storePrevIter<tensor>();
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::solutionControlx::solutionControlx(fvMesh& mesh, const word& algorithmName)
:
    IOobject
    (
        "solutionControlx",
        mesh.time().timeName(),
        mesh
    ),
    mesh_(mesh),
    residualControl_(),
    algorithmName_(algorithmName),
    nNonOrthCorr_(0),
    nMonitorIter_(1),
    convFac_(1.0),
    momentumPredictor_(true),
    transonic_(false),
    corr_(0),
    corrNonOrtho_(0)
{}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::solutionControlx::~solutionControlx()
{}


// ************************************************************************* //
