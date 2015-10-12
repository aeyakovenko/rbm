{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
module Data.ImageUtils(appendGIF
                      ,writeBMP
                      ) where

import Control.Applicative((<|>))
import qualified Data.Matrix as M
import qualified Data.Array.Repa as R
import qualified Data.Array.Repa.IO.BMP as R
import qualified Data.Array.Repa.Algorithms.Pixel as R
import Codec.Picture.Gif as G
import Codec.Picture.Types as G
import qualified Data.ByteString as BS
import Data.Matrix(Matrix(..), U, B, I)
import qualified Data.Vector.Storable as VS

generateBox::Monad m => Matrix U B I -> m (Matrix U B B)
generateBox mm@(Matrix bxi) = do
   let
       imagewidth :: Int
       imagewidth = round $ (fromIntegral $ M.col mm)**(0.5::Double)
       batches = M.row mm 
       pixels = M.col mm - 1
       computeImage (R.Z R.:. brix R.:. bcix) = 
         let  (imagenum,imagepixel) = index batches pixels brix bcix
              pos =  R.Z R.:. imagenum R.:. (imagepixel + 1)
              safeIndex m (R.Z R.:.mr R.:.mc) (R.Z R.:.r R.:. c)
                  | mr <= r || mc <= c = 0
                  | otherwise = m R.! (R.Z R.:.r R.:.c)
         in    safeIndex bxi (R.extent bxi) pos
       imagesperside = ceiling $ (fromIntegral $ M.row mm)**(0.5::Double) 
       sh = R.Z R.:. imagewidth * imagesperside R.:. imagewidth * imagesperside
   image <- Matrix <$> (R.computeUnboxedP $ R.fromFunction sh computeImage)
   return image

index :: Int -> Int -> Int -> Int -> (Int,Int)
index batches pixels rr cc = (image, pixel) 
   where imagewidth = round $ (fromIntegral pixels)**(0.5::Double)
         imagesperside = ceiling $ (fromIntegral batches)**(0.5::Double) 
         image = (rr `div` imagewidth) * imagesperside + (cc `div` imagewidth)
         pixel = (rr `mod` imagewidth) * imagewidth + (cc `mod` imagewidth)

checkE :: Either String t -> t
checkE (Left err) = error err
checkE (Right a) = a

toImage :: Matrix U B B -> G.Image G.Pixel8
toImage img = G.Image (M.row img) (M.col img) $ VS.fromList $ map toPixel $ M.toList img
   where toPixel xx = round $ 255 * xx

appendGIF:: String -> Matrix U B I -> IO ()
appendGIF sfile mm' = do
   mm <- generateBox mm' 
   let check (Left err) = error err
       check (Right a) = a
       fromDynamic (G.ImageRGB8 im) = (G.greyPalette, 10,  G.extractComponent G.PlaneRed im)
       fromDynamic _  = error "unexpected image type"
   images <- (map fromDynamic <$> check <$> G.decodeGifImages <$> (BS.readFile sfile)) 
         <|> (return [])
   let img = (G.greyPalette, 10, toImage mm)
   putStrLn $ concat ["writing image: ", sfile]
   checkE $ G.writeGifImages sfile G.LoopingForever (images ++ [img])

writeBMP::String -> Matrix U B I -> IO ()
writeBMP sfile bxi = do
   (Matrix image) <- generateBox bxi
   ar <- R.computeUnboxedP $ R.map R.rgb8OfGreyDouble image
   putStrLn $ concat ["writing image: ", sfile]
   R.writeImageToBMP sfile ar


