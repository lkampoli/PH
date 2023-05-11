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

Class
    Foam::solutionControlx

Description
    Base class for solution control classes

\*---------------------------------------------------------------------------*/

#ifndef solutionControlx_H
#define solutionControlx_H

#include "fvMesh.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                       Class solutionControlx Declaration
\*---------------------------------------------------------------------------*/

class solutionControlx
:
    public IOobject
{
public:

    struct fieldData
    {
        wordRe name;
        scalar absTol;
        scalar relTol;
        scalar initialResidual;
        List<scalar> lastResiduals;
    };


protected:

    // Protected data

        //- Reference to the mesh database
        fvMesh& mesh_;

        //- List of residual data per field
        List<fieldData> residualControl_;

        //- The dictionary name, e.g. SIMPLE, PIMPLE
        const word algorithmName_;


        // Solution controls

            //- Maximum number of non-orthogonal correctors
            label nNonOrthCorr_;

            //- Number of iterations to monitor for convergence
            label nMonitorIter_;

            //- Convergence factor to bind residual around average
            scalar convFac_;

            //- Flag to indicate to solve for momentum
            bool momentumPredictor_;

            //- Flag to indicate to solve using transonic algorithm
            bool transonic_;


        // Evolution

            //- Current corrector loop index
            label corr_;

            //- Current non-orthogonal corrector loop index
            label corrNonOrtho_;


    // Protected Member Functions

        //- Read controls from fvSolution dictionary
        virtual void read(const bool absTolOnly);

        //- Return index of field in residualControl_ if present
        virtual label applyToField
        (
            const word& fieldName,
            const bool useRegEx = true
        ) const;

        //- Return true if all convergence checks are satisfied
        virtual bool criteriaSatisfied() = 0;

        //- Store previous iteration fields
        virtual void storePrevIterFields() const;

        //- Store previous iteration field for vol<Type>Fields
        template<class Type>
        void storePrevIter() const;

        //- Disallow default bitwise copy construct
        solutionControlx(const solutionControlx&);

        //- Disallow default bitwise assignment
        void operator=(const solutionControlx&);


public:


    // Static Data Members

        //- Run-time type information
        TypeName("solutionControlx");


    // Constructors

        //- Construct from mesh
        solutionControlx(fvMesh& mesh, const word& algorithmName);


    //- Destructor
    virtual ~solutionControlx();


    // Member Functions

        // Access

            //- Return the solution dictionary
            inline const dictionary& dict() const;

            //- Current corrector loop index
            inline label corr() const;

            //- Current non-orthogonal corrector index
            inline label corrNonOrtho() const;


        // Solution control

            //- Maximum number of non-orthogonal correctors
            inline label nNonOrthCorr() const;

            //- Helper function to identify final non-orthogonal iteration
            inline bool finalNonOrthogonalIter() const;

            //- Flag to indicate to solve for momentum
            inline bool momentumPredictor() const;

            //- Flag to indicate to solve using transonic algorithm
            inline bool transonic() const;


        // Evolution

            //- Main control loop
            virtual bool loop() = 0;

            //- Non-orthogonal corrector loop
            inline bool correctNonOrthogonal();
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#ifdef NoRepository
    #include "solutionControlxTemplates.C"
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#include "solutionControlxI.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
