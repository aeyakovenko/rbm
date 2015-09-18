{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE ExistentialQuantification #-}
module Data.Matrix( Matrix(..)
                  , MatrixOps(..)
                  , R.U
                  , R.D
                  ) where

import qualified Data.Array.Repa as R
import qualified Data.Array.Repa.Algorithms.Matrix as R
import qualified Data.Array.Repa.Unsafe as Unsafe
import Control.DeepSeq(NFData, rnf)
import Data.Array.Repa(Array
                      ,U
                      ,D
                      ,DIM2
                      ,Any(Any)
                      ,Z(Z)
                      ,(:.)((:.))
                      ,All(All)
                      )

data Matrix d a b = R.Source d Double => Matrix (Array d DIM2 Double)

instance NFData (Matrix U a b) where
   rnf (Matrix ar) = ar `R.deepSeqArray` ()

class MatrixOps a b where
   mmult :: Monad m => (Matrix U a b) -> (Matrix U b a) -> m (Matrix U a a)
   mmultT :: Monad m => (Matrix U a b) -> (Matrix U a b) -> m (Matrix U a a)
   d2u :: Monad m => Matrix D a b -> m (Matrix U a b)
   (*^) :: Matrix c a b -> Matrix d a b -> (Matrix D a b)
   (+^) :: Matrix c a b -> Matrix d a b -> (Matrix D a b)
   (-^) :: Matrix c a b -> Matrix d a b -> (Matrix D a b)
   map :: (Double -> Double) -> Matrix c a b -> (Matrix D a b)

instance MatrixOps a b where
   mmult (Matrix ab) (Matrix ba) = Matrix <$> (ab `mmultP` ba)
   mmultT (Matrix ab) (Matrix ab') = Matrix <$> (ab `mmultTP` ab')
   d2u (Matrix ar) = Matrix <$> (R.computeP ar)
   (Matrix ab) *^ (Matrix ab') = Matrix (ab R.*^ ab')
   (Matrix ab) +^ (Matrix ab') = Matrix (ab R.+^ ab')
   (Matrix ab) -^ (Matrix ab') = Matrix (ab R.-^ ab')
   map f (Matrix ar) = Matrix (R.map f ar)

{--
 - matrix multiply
 - a x (transpose b)
 - based on mmultP from repa-algorithms-3.3.1.2
 -}
mmultTP  :: Monad m
        => Array U DIM2 Double
        -> Array U DIM2 Double
        -> m (Array U DIM2 Double)
mmultTP arr trr
 = [arr, trr] `R.deepSeqArrays`
   do
        let (Z :. h1  :. _) = R.extent arr
        let (Z :. w2  :. _) = R.extent trr
        R.computeP
         $ R.fromFunction (Z :. h1 :. w2)
         $ \ix   -> R.sumAllS
                  $ R.zipWith (*)
                        (Unsafe.unsafeSlice arr (Any :. (R.row ix) :. All))
                        (Unsafe.unsafeSlice trr (Any :. (R.col ix) :. All))
{-# NOINLINE mmultTP #-}

{--
 - regular matrix multiply
 - a x b
 - based on mmultP from repa-algorithms-3.3.1.2
 - basically moved the deepseq to seq the trr instead of brr
 -}
mmultP  :: Monad m
        => Array U DIM2 Double
        -> Array U DIM2 Double
        -> m (Array U DIM2 Double)
mmultP arr brr
 = do   trr <- R.transpose2P brr
        mmultTP arr trr

